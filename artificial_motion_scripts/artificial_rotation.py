from pydicom import dcmread
import SimpleITK as sitk
import os
import math
import argparse
import numpy as np
import subprocess
from scipy.spatial.transform import Rotation



def apply_rotation(args, dir, extension):
    """
    Resample volume, creating a copy of the reference volume rotated in a sinusoidal fashion

    param args: input args
    param dir: directory to write volumes
    param extensions: file extension of input volume
    """

    if os.path.exists('parameters.csv'):
        os.remove('parameters.csv')

    f = open('parameters.csv', 'w') #open in write mode

    try:
        reference = sitk.ReadImage(args.inVol) #read in vol
    except:
        print("Input Image is not Valid") #throw an error otherwise
        exit(1)

    # Read metadata
    reader = sitk.ImageFileReader()
    reader.SetFileName(args.inVol)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    # Center of volume.
    reference_center = reference.TransformContinuousIndexToPhysicalPoint(
        [(index-1)/2.0 for index in reference.GetSize()] )

    for i in range(args.num_vols):        
        sin = math.sin(0.2*i) 
        angle_x = math.radians(args.angle_x) * sin
        angle_y = math.radians(args.angle_y) * sin
        angle_z = math.radians(args.angle_z) * sin
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
        transform = sitk.AffineTransform(3)
        transform.SetMatrix(np.ravel(combined_rotation))
        transform.SetCenter(reference_center)

        # Perform resampling 
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetOutputDirection(reference.GetDirection())
        resampler.SetOutputOrigin(reference.GetOrigin())
        resampler.SetOutputSpacing(reference.GetSpacing())
        resampler.SetSize(reference.GetSize())
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetDefaultPixelValue(0.0)
        resampler.SetOutputPixelType(reference.GetPixelID())

        transformed_image = resampler.Execute(reference)

        # Copy over as much metadata as possible
        for j in (reader.GetMetaDataKeys()):
            transformed_image.SetMetaData(j, reader.GetMetaData(j))

        sitk.WriteImage(transformed_image, os.path.join(dir, f'rotated_{str(i).zfill(4)}{extension}'))
        print(f'Volume {i} Rotation: Rotation X: {math.degrees(angle_x)} Degrees, Rotation Y: {math.degrees(angle_y)} Degrees, Rotation Z: {math.degrees(angle_z)} Degrees')

        # Change rotation parameters from euler to versor since sms-mi-reg is in versor
        rot = Rotation.from_euler('xyz', (angle_x, angle_y, angle_z), degrees=False) #REMEMBER TO CHANGE THE DEGREES FLAG IF NEEDED
        rot_quat = rot.as_quat()

        # Plot simulated motion transformations for a reference plot, only if the flag is used
        write_simulated_data(f, args, list(rot_quat[0:3]), i, reference_center)

    f.close()


def metadata(inVol, indirectory):
    '''
    This function handles metadata generation of the artificial volumes generated. This has to be done because SITK does not handle the metadata fields well at all, and these specific fields are necessary for SLIMM
    to function properly. More specifically, if you want to convert from dicom images to an MRD image for SLIMM these fields are needed. The dicom2mrd script will handle the rest

    param inVol: reference volume that is used to populate the artificial image metadata
    param indirectory: directory of artificial images
    '''
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
    '''
    Perform Volume-to-Volume registration, using first volume (which has 0 motion) as reference. Uses previous transform file as input transform file. For example, vol 2 would use vol 1 transform file.
    Vol 0 is used as reference, Vol 1 is registered from identity transform, Vol 2 uses Vol 1 transform file result as input, etc.

    param refVol: reference volume
    param voldir: output directory for volumes
    '''
    # list all volume files and sort them in order
    files = os.listdir(voldir)
    files.sort()

    # host computer : container directory alias
    dirmapping = os.getcwd() + ":" + "/data"
    dockerprefix = ["docker","run","--rm", "-it", "--init", "-v", dirmapping,
        "--user", str(os.getuid())+":"+str(os.getgid())]

    inputTransformFileName = "identity-centered.tfm"
    subprocess.run( dockerprefix +  [ "crl/sms-mi-reg", "crl-identity-transform-at-volume-center.py", 
    "--refvolume", refVol, 
    "--transformfile", inputTransformFileName ] )

    # Perform vvr with volumes, looping through until all volumes have been registered
    for i, volname in enumerate(files):
        vol = os.path.join(voldir, volname)
        outTransFile = str(i).zfill(4)

        if i == 0: # Set variable for first, unmoved volume and go to next iteration
            firstVol = vol
        elif i == 1: # Second volume uses identity transform
            subprocess.run( dockerprefix + ["crl/sms-mi-reg", "sms-mi-reg", 
            firstVol,
            inputTransformFileName,
            outTransFile,
            vol] )
        else:  # Every volume after uses previous volume tfm
            inputTransformFileName = f"sliceTransform{str(i - 1).zfill(4)}.tfm"

            subprocess.run( dockerprefix + ["crl/sms-mi-reg", "sms-mi-reg", 
            firstVol,
            inputTransformFileName,
            outTransFile,
            vol] )


def write_simulated_data(f, args, rot, i, center):
    '''
    Function that writes the 6 simulated motion parameters to a csv file and creates transform files for each slice or set of slices, all to create a reference plot, if the refplot flag is active

    param f: csv file
    param args: input args
    param rot: rotation parameters
    param i: slice index to write transform file name
    param center: fixed parameters to set center of rotation
    '''
    # Create 6 parameter array to write to csv file
    transform_array = rot + (0,0,0)
    a = np.asarray(transform_array).reshape(1,-1)
    np.savetxt(f, a, delimiter=",",fmt="%.8f")
    
    # Create transform files using artificial parameters and run motion monitor to generate plot
    if args.refplot:
        transform1 = sitk.VersorRigid3DTransform()
        transform1.SetCenter(center) #center of rotation
        transform1.SetParameters(transform_array)
        sitk.WriteTransform(transform1, f'./{str(i).zfill(4)}.tfm')

        if i == (args.num_vols - 1):
            dirmapping_a = os.getcwd() + ":" + "/data"
            dockerprefix_a = ["docker","run","--rm", "-it", "--init", "-v", dirmapping_a, "--user", str(os.getuid())+":"+str(os.getgid())]

            subprocess.run( dockerprefix_a + ["jauger/motion-monitor:latest", "0000.tfm"])
            # Loop through files in the directory
            for filename in os.listdir(os.getcwd()):
                if filename.endswith(".tfm"):
                    os.remove(filename)


if __name__ == '__main__':
    '''
    Simple script that only rotates image in sinusoidal fashion. Option to perform VVR.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-inVol', default='./input/adultjosh.dcm', help='input folder with single dicom file that is duplicated')
    
    parser.add_argument('-num_vols', default=40, type=int, help='number of output dicoms')
    
    parser.add_argument('-angle_x', default=0, type=float, help='maximum angle of rotation in degrees x-axis (roll)')
    parser.add_argument('-angle_y', default=0, type=float, help='maximum angle of rotation in degrees, y-axis (pitch)')
    parser.add_argument('-angle_z', default=0, type=float, help='maximum angle of rotation in degrees, z-axis (yaw)')

    parser.add_argument('--vvr', action='store_true', help='flag for performing volume to volume registration')
    parser.add_argument('--refplot', action='store_true', help='flag for creating plot of simulated motion')
    parser.add_argument('--slimm', action='store_true', help='flag to perform metadata regeneration on dicom images to use on slimm')

    args = parser.parse_args()

    # Create output directory if it does not exist yet
    directory = "rotated_vols"
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get file extension for saving purporses
    extension = os.path.splitext(args.inVol)
    
    # Apply artificial rotation
    apply_rotation(args, directory, extension[1])
    
    # Only do metadata regeneration step if the file is a dicom
    if extension[1] == '.dcm' and args.slimm:
        metadata(args.inVol, directory)

    # Perform vvr and motion monitor if the flag is used
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