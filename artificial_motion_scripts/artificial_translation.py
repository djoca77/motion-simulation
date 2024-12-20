from pydicom import dcmread
import SimpleITK as sitk
import os
import math
import argparse
import numpy as np
import subprocess



def apply_translation(args, dir, extension):
    """
    Resample volume, creating a copy of the reference volume translated in a sinusoidal fashion

    param args: input args
    param dir: directory to write volumes
    param extensions: file extension of input volume
    """
    if os.path.exists('parameters.csv'):   
        os.remove('parameters.csv')
    f = open('parameters.csv', 'w') # reopen in append mode

    try:
        reference = sitk.ReadImage(args.inVol) #read in vol
    except:
        print("Input Image is not Valid") #throw an error otherwise
        exit(1)

    # Read Metadata
    reader = sitk.ImageFileReader()
    reader.SetFileName(args.inVol)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    # Center of volume.
    reference_center = reference.TransformContinuousIndexToPhysicalPoint(
        [(index-1)/2.0 for index in reference.GetSize()] )

    for i in range(args.num_vols):
        # Set translation array
        sin = math.sin((args.period / args.num_vols) * i)
        translation_array = ((args.x * sin), (args.y * sin), (args.z * sin))
        transform = sitk.AffineTransform(3)
        transform.SetTranslation(translation_array)

        # Resample volume using translation transform
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetOutputOrigin(reference.GetOrigin())
        resampler.SetOutputDirection(reference.GetDirection())
        resampler.SetOutputSpacing(reference.GetSpacing())
        resampler.SetSize(reference.GetSize())
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetDefaultPixelValue(0.0) 
        resampler.SetOutputPixelType(reference.GetPixelID())
        transformed_image = resampler.Execute(reference)

        # Regenerate metadata fields manually
        for j in (reader.GetMetaDataKeys()):
            transformed_image.SetMetaData(j, reader.GetMetaData(j))

        sitk.WriteImage(transformed_image, os.path.join(dir, f'translated_{str(i).zfill(4)}{extension}'))  #write slice to directory
        print(f'Volume {i} Translation: Translation X: {translation_array[0]} mm, Translation Y: {translation_array[1]} mm, Translation Z: {translation_array} mm')

        # Plot simulated motion transformations for a reference plot, only if the flag is used
        write_simulated_data(f, args, translation_array, i, reference_center)

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
            # dicom.PerFrameFunctionalGroupsSequence._list[k].PlaneOrientationSequence._list[0].ImageOrientationPatient._list = ['1','0','0','0','1','0']

        # Save the new DICOM file
        dicom_dir = 'slimm'
        if not os.path.exists(dicom_dir):
            os.makedirs(dicom_dir)
        path = os.path.join(dicom_dir, f'slimm_{i}.dcm')
        dicom.save_as(path)
    print("Metadata Augmentation Complete")


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
        else: # Every volume after uses previous volume tfm
            inputTransformFileName = f"sliceTransform{str(i - 1).zfill(4)}.tfm"

            subprocess.run( dockerprefix + ["crl/sms-mi-reg", "sms-mi-reg", 
            firstVol,
            inputTransformFileName,
            outTransFile,
            vol] )

def write_simulated_data(f, args, translation_array, i, center):
    '''
    Function that writes the 6 simulated motion parameters to a csv file and creates transform files for each slice or set of slices, all to create a reference plot, if the refplot flag is active

    param f: csv file
    param args: input args
    param rot: rotation parameters
    param i: slice index to write transform file name
    param center: fixed parameters to set center of rotation
    '''
    # Write to csv file the 6 parameters
    transform_array =(0,0,0) + translation_array
    a = np.asarray(transform_array).reshape(1,-1)
    np.savetxt(f, a, delimiter=",",fmt="%.8f")
    
    # Run motion monitor for artificial transform files once resampling on desired number of volumes is complete
    if args.refplot:
        transform1 = sitk.VersorRigid3DTransform()
        transform1.SetCenter(center) #center of rotation #COME BACK TO THIS
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
    Simple script that performs artificial translation in all 3 axes in a sinusoidal fashion. Option to perform VVR
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-inVol', default='./input/adultjosh.dcm', help='file path of input volume to perform artificial translation') #change this to inVol instead of -inVol. same for x y z vals

    parser.add_argument('-num_vols', default=40, type=int, help='number of output dicoms, default is 40. Number must be integer')

    parser.add_argument('-x', default=2.5, type=float, help='x-axis translation sin wave magnitude. default is 0. Number can be integer or decimal')
    parser.add_argument('-y', default=0, type=float, help='y-axis translation sin wave magnitude. default is 0. Number can be integer or fldecimaloat')
    parser.add_argument('-z', default=0, type=float, help='z-axis translation sin wave magnitude. default is 0. Number can be integer or decimal')
    
    parser.add_argument('-period', default=2*math.pi, help='period of sinusoidal motion, default is 0.2. Number can be integer or decimal') 

    parser.add_argument('--vvr', action='store_true', help='flag for performing volume to volume registration')
    parser.add_argument('--refplot', action='store_true', help='flag to create motion monitor plots for simulated motion parameters')
    parser.add_argument('--slimm', action='store_true', help='flag to perform metadata regeneration on dicom images to use on slimm')

    args = parser.parse_args()

    # create output directory if it does not exist yet
    directory = "translated_vols"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # extract file extension and file name for proper file writing
    extension = os.path.splitext(args.inVol)

    # apply simulated translation  
    apply_translation(args, directory, extension[1])

    #only do metadata regeneration step if the file is a dicom, so that it can be converted to an mrd for slimm
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