from pydicom import dcmread
import SimpleITK as sitk
import os
import math
import argparse
import numpy as np
import subprocess
import math

def apply_rotation(args, dir, extension):
    """
    Resample volume, creating a copy of the reference volume rotated in a sinusoidal fashion
    """

    if os.path.exists('parameters.csv'):
        os.remove('parameters.csv')

    f = open('parameters.csv', 'w') #open in write mode

    try:
        reference = sitk.ReadImage(args.inVol) #read in vol
    except:
        print("Input Image is not Valid") #throw an error otherwise
        exit(1)

    #metadata
    reader = sitk.ImageFileReader()
    reader.SetFileName(args.inVol)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    for i in range(int(args.num_dicoms)):        
        # Center of volume.
        reference_center = reference.TransformContinuousIndexToPhysicalPoint(
            [(index-1)/2.0 for index in reference.GetSize()] )

        transform = sitk.AffineTransform(3)
        
        sin = math.sin(0.2*i) 
        angle_x = math.radians(float(args.angle_x)) * sin
        angle_y = math.radians(float(args.angle_y)) * sin
        angle_z = math.radians(float(args.angle_z)) * sin
        # Rotation matrix around the X-axis
        rotation_x = [[1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)]]

        # Rotation matrix around the Y-axis
        rotation_y = [[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]]

        # Rotation matrix around the Z-axis
        rotation_z = [[np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1]]

        # Combine rotations: R = Rz * Ry * Rx (order matters)
        combined_rotation = np.dot(rotation_z, np.dot(rotation_y, rotation_x))
        transform.SetMatrix(np.ravel(combined_rotation))
        
        transform.SetCenter(reference_center)

        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetOutputDirection(reference.GetDirection())
        resampler.SetOutputOrigin(reference.GetOrigin())
        resampler.SetOutputSpacing(reference.GetSpacing())
        resampler.SetSize(reference.GetSize())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)

        transformed_image = resampler.Execute(reference)

        for j in (reader.GetMetaDataKeys()):
            transformed_image.SetMetaData(j, reader.GetMetaData(j))

        sitk.WriteImage(transformed_image, os.path.join(dir, f'rotated_{str(i).zfill(4)}{extension[1]}'))
        print(f'Volume {i} Rotation: Rotation X: {math.degrees(angle_x)} Degrees, Rotation Y: {math.degrees(angle_y)} Degrees, Rotation Z: {math.degrees(angle_z)} Degrees')

        # plot simulated motion transformations for a reference plot, only if the flag is used
        write_simulated_data(f, args, (angle_x, angle_y, angle_z), i, reference_center)

    f.close()
        
def metadata(inVol, indirectory):
    files = os.listdir(indirectory)
    files.sort()
    for i, file in enumerate(files):        
        reference = dcmread(inVol)
        dicom = dcmread(os.path.join(indirectory, file))
        dicom.InstanceNumber = i + 1
        for k in range(int(dicom.NumberOfFrames)):
            dicom.PerFrameFunctionalGroupsSequence._list[k].add_new([0x0020, 0x9111], 'SQ', reference.PerFrameFunctionalGroupsSequence._list[k].FrameContentSequence)
            dicom.PerFrameFunctionalGroupsSequence._list[k].add_new([0x0018, 0x9114], 'SQ', reference.PerFrameFunctionalGroupsSequence._list[k].MREchoSequence)
            dicom.SharedFunctionalGroupsSequence._list[0].add_new([0x0018, 0x9112], 'SQ', reference.SharedFunctionalGroupsSequence._list[0].MRTimingAndRelatedParametersSequence)
            dicom.PerFrameFunctionalGroupsSequence._list[k].add_new([0x0028, 0x9110], 'SQ', reference.PerFrameFunctionalGroupsSequence._list[k].PixelMeasuresSequence)
            dicom.SharedFunctionalGroupsSequence._list[0].add_new([0x0018, 0x9115], 'SQ', reference.SharedFunctionalGroupsSequence._list[0].MRModifierSequence)
            dicom.PerFrameFunctionalGroupsSequence._list[k].add_new([0x0020, 0x9116], 'SQ', reference.PerFrameFunctionalGroupsSequence._list[k].PlaneOrientationSequence)
            dicom.PerFrameFunctionalGroupsSequence._list[k].PlaneOrientationSequence._list[0].ImageOrientationPatient._list = ['1','0','0','0','1','0']

        # Save the new DICOM file
        dicom_dir = 'slimm'
        if not os.path.exists(dicom_dir):
            os.makedirs(dicom_dir)
        path = os.path.join(dicom_dir, f'slimm_{i}.dcm')
        dicom.save_as(path)

def vvr(refVol, voldir):
    # list all volume files and sort them in order
    files = os.listdir(voldir)
    files.sort()

    print('refVolumeName is ', str(refVol) )

    # host computer : container directory alias
    dirmapping = os.getcwd() + ":" + "/data"
    dockerprefix = ["docker","run","--rm", "-it", "--init", "-v", dirmapping,
        "--user", str(os.getuid())+":"+str(os.getgid())]
    print(dockerprefix)

    inputTransformFileName = "identity-centered.tfm"
    subprocess.run( dockerprefix +  [ "crl/sms-mi-reg", "crl-identity-transform-at-volume-center.py", 
    "--refvolume", refVol, 
    "--transformfile", inputTransformFileName ] )

    # Perform vvr with volumes, looping through until all volumes have been registered
    for i, volname in enumerate(files):
        vol = os.path.join(voldir, volname)
        outTransFile = str(i).zfill(4)

        if i == 0: #use identity-centered.tfm for only the first registration
            subprocess.run( dockerprefix + ["crl/sms-mi-reg", "sms-mi-reg", 
            refVol,
            inputTransformFileName,
            outTransFile,
            vol] )
        else: #use the most recent written transform file for the next registration
            inputTransformFileName = f"sliceTransform{str(i - 1).zfill(4)}.tfm"

            print( dockerprefix + ["crl/sms-mi-reg", "sms-mi-reg", 
            refVol,
            inputTransformFileName,
            outTransFile,
            vol] )

            subprocess.run( dockerprefix + ["crl/sms-mi-reg", "sms-mi-reg", 
            refVol,
            inputTransformFileName,
            outTransFile,
            vol] )


def write_simulated_data(f, args, theta, i, center):
        #resample slice using translation transform
        transform_array = theta + (0,0,0)
        a = np.asarray(transform_array).reshape(1,-1)
        np.savetxt(f, a, delimiter=",",fmt="%.8f")
        
        if args.refplot:
            transform1 = sitk.VersorRigid3DTransform()
            transform1.SetCenter(center) #center of rotation
            transform1.SetParameters(transform_array)
            sitk.WriteTransform(transform1, f'./{str(i).zfill(4)}.tfm')

            if i == (args.num_dicoms - 1):
                dirmapping_a = os.getcwd() + ":" + "/data"
                dockerprefix_a = ["docker","run","--rm", "-it", "--init", "-v", dirmapping_a, "--user", str(os.getuid())+":"+str(os.getgid())]

                subprocess.run( dockerprefix_a + ["jauger/motion-monitor:latest", "0000.tfm"])
                # Loop through files in the directory
                for filename in os.listdir(os.getcwd()):
                    if filename.endswith(".tfm"):
                        os.remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-inVol', default='./input/adultjosh.dcm', help='input folder with single dicom file that is duplicated')
    
    parser.add_argument('-num_dicoms', default=40, help='number of output dicoms')
    parser.add_argument('-angle_x', default=0, help='maximum angle of rotation in degrees x-axis (roll)')
    parser.add_argument('-angle_y', default=0, help='maximum angle of rotation in degrees, y-axis (pitch)')
    parser.add_argument('-angle_z', default=20, help='maximum angle of rotation in degrees, z-axis (yaw)')

    parser.add_argument('--vvr', action='store_true', help='flag for performing volume to volume registration')
    parser.add_argument('--refplot', action='store_true', help='flag for creating plot of simulated motion')
    parser.add_argument('--slimm', action='store_true', help='flag to perform metadata regeneration on dicom images to use on slimm')

    args = parser.parse_args()

    # create output directory if it does not exist yet
    directory = "rotated_vols"
    if not os.path.exists(directory):
        os.makedirs(directory)

    #get file extension for saving purporses
    extension = os.path.splitext(args.inVol)
    
    # apply artificial rotation
    apply_rotation(args, directory, extension)
    
    #only do metadata regeneration step if the file is a dicom
    if extension[1] == '.dcm' and args.slimm:
        metadata(args.inVol, directory)

    #perform vvr and motion monitor if the flag is used
    if args.vvr:
        vvr(args.inVol, directory)
    
        # Motion monitor
        os.remove("identity-centered.tfm")
        dirmapping_a = os.getcwd() + ":" + "/data"
        dockerprefix_a = ["docker","run","--rm", "-it", "--init", "-v", dirmapping_a, "--user", str(os.getuid())+":"+str(os.getgid())]

        subprocess.run( dockerprefix_a + ["jauger/motion-monitor:latest", "sliceTransform0000.tfm"])
        # Loop through files in the directory
        for filename in os.listdir(os.getcwd()):
            if filename.endswith(".tfm"):
                os.remove(filename)

