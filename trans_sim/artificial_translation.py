import SimpleITK as sitk
import os
import math
import argparse
import numpy as np
from pydicom import dcmread
import subprocess

def apply_translation(args, dir, extension):
    """
    Resample volume, creating a copy of the reference volume translated in a sinusoidal fashion
    """
    if os.path.exists('parameters.csv'):   
        os.remove('parameters.csv')
    f = open('parameters.csv', 'w') # reopen in append mode

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
        #set translation array
        sin = math.sin(float(args.period)*i)
        translation_array = ((args.x * sin), (args.y * sin), (args.z * sin))
        transform = sitk.AffineTransform(3)
        transform.SetTranslation(translation_array)

        #resample volume using translation transform
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetOutputOrigin(reference.GetOrigin())
        resampler.SetOutputDirection(reference.GetDirection())
        resampler.SetOutputSpacing(reference.GetSpacing())
        resampler.SetSize(reference.GetSize())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0) 
        resampler.SetOutputPixelType(reference.GetPixelID())
        transformed_image = resampler.Execute(reference)

        #regenerate metadata fields manually
        for j in (reader.GetMetaDataKeys()):
            transformed_image.SetMetaData(j, reader.GetMetaData(j))

        sitk.WriteImage(transformed_image, os.path.join(dir, f'translated_{str(i).zfill(4)}{extension[1]}'))  #write slice to directory
        print(f'Volume {i} Translation: Translation X: {translation_array[0]} mm, Translation Y: {translation_array[1]} mm, Translation Z: {translation_array} mm')

        # plot simulated motion transformations for a reference plot, only if the flag is used
        write_simulated_data(f, args, translation_array, i)

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
            # dicom.PerFrameFunctionalGroupsSequence._list[k].PlaneOrientationSequence._list[0].ImageOrientationPatient._list = ['1','0','0','0','1','0']

        # Save the new DICOM file
        dicom_dir = 'slimm'
        if not os.path.exists(dicom_dir):
            os.makedirs(dicom_dir)
        path = os.path.join(dicom_dir, f'slimm_{i}.dcm')
        dicom.save_as(path)
    print("Metadata Augmentation Complete")

def vvr(refVol, voldir):
    # list all volume files and sort them in order
    files = os.listdir(voldir)
    files.sort()

    print('refVolumeName is ', str(refVol) )

    # host computer : container directory alias
    dirmapping = os.getcwd() + ":" + "/data"
    dockerprefix = ["docker","run","--rm", "-it", "--init", "-v", dirmapping,
        "--user", str(os.getuid())+":"+str(os.getgid())]
    
    inputTransformFileName = "identity-centered.tfm"
    subprocess.run( dockerprefix +  [ "crl/sms-mi-reg", "crl-identity-transform-at-volume-center.py", 
    "--refvolume", refVol, 
    "--transformfile", inputTransformFileName ] )

    # Perform vvr on volumes, using first, unmoved volume as the reference
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

def write_simulated_data(f, args, translation_array, i):
        #write to csv file the 6 parameters
        transform_array =(0,0,0) + translation_array
        a = np.asarray(transform_array).reshape(1,-1)
        np.savetxt(f, a, delimiter=",",fmt="%.8f")
        
        if args.refplot:
            transform1 = sitk.VersorRigid3DTransform()
            transform1.SetCenter((0,0,0)) #center of rotation #COME BACK TO THIS
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

    parser.add_argument('-inVol', default='./input/adultjosh.dcm', help='file path of input volume to perform artificial translation') #change this to inVol instead of -inVol. same for x y z vals

    parser.add_argument('-x', default=2.5, help='x-axis translation sin wave magnitude. default is 0. Number can be integer or decimal')
    parser.add_argument('-y', default=0, help='y-axis translation sin wave magnitude. default is 0. Number can be integer or fldecimaloat')
    parser.add_argument('-z', default=0, help='z-axis translation sin wave magnitude. default is 0. Number can be integer or decimal')
    parser.add_argument('-period', default=0.2, help='period of sinusoidal motion, default is 0.2. Number can be integer or decimal') 
    parser.add_argument('-num_dicoms', default=40, help='number of output dicoms, default is 40. Number must be integer')

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
    apply_translation(args, directory, extension)

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