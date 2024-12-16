import SimpleITK as sitk
import os
import math
import argparse
import numpy as np
import subprocess
from pydicom import dcmread
from scipy.spatial.transform import Rotation

        
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


def interleaved_array(size, interleaved_factor):
    interleaved_array = []
    current_value = 0
    base_value = 0

    while interleaved_factor > base_value:
        if current_value < size:
            interleaved_array.append(current_value)
            current_value += interleaved_factor
        else:
            # Set the starting point for the next cycle
            base_value += 1
            current_value = base_value
        print(current_value)

    return interleaved_array


def aquisition_time(args):
    dcm = dcmread(args.invol)
    uSliceTime = np.empty(len(dcm.PerFrameFunctionalGroupsSequence))
    for iSlice in range(len(dcm.PerFrameFunctionalGroupsSequence)):
        uSliceTime[iSlice] = float(dcm.PerFrameFunctionalGroupsSequence[iSlice].FrameContentSequence[0].FrameAcquisitionDateTime) 

    _, counts = np.unique(uSliceTime, return_counts=True)
    sms = counts.max()
    
    sortedSliceTime = [i for i, _ in sorted(enumerate(uSliceTime), key=lambda x: x[1])]

    return sortedSliceTime, sms


def apply_translation(args, dir, extension, iter):
    '''
    Resample volume in an interleaved manner, such that it follows a pattern like this for example: [0,20,1,21,2,22,3,23...]
    The output files are numbered in order, but correspond to a different slice than the number of the file. For example, 
    slice_0000 is for slice 0 positionally and slice_0001 is for slice 20 positionally based on the interleaving
    This is for ease of inputting the files into sms-mi-reg
    '''
    try:
        reference = sitk.ReadImage(args.invol) #read in vol
    except:
        print("Input Image is not Valid") #throw an error otherwise
        exit(1)
    size = reference.GetSize() #returns 3D size array
    num_slices = int(size[2])

    if os.path.exists('parameters.csv'):   
        os.remove('parameters.csv') #remove
    f = open('parameters.csv', 'w') #reopen in write mode

    transformed_image = sitk.Image(reference) #create a copy of volume to modify with slice motion
    sms = args.sms_factor

    if extension[1] == '.dcm':
        indices, sms = aquisition_time(args) #creates array of indexes based on the aquisition time of the slices to create a better simulation of the data
    elif args.interleaved:
        indices = interleaved_array(num_slices, args.interleaved_factor) #creates interleaved array depending on sms factor that determines the order of slices
    else:
        indices = list(range(num_slices))
    
    print(f'Order of slices: {indices}')

    for i, idx in enumerate(indices): #for every slice in the volume
        #extract slice
        startIndex = (0, 0, idx)
        sizeROI = (size[0], size[1], 1)
        image = sitk.RegionOfInterest(reference, sizeROI, startIndex)
        

        #create 3D array of sinusoidal motion to be applied translationally to slice
        sin = math.sin(( 2 * math.pi / ( num_slices /  sms )) * ( i // sms)) #// int(args.sms)) FOR SIMULTANEOUS ACQUISITION SIMULATION TAKES THE FLOOR SO IT GOES 0 0 1 1 2 2 3 3 ETC
        translation_array = ((args.x * sin), (args.y * sin), 0)
        transform = sitk.AffineTransform(3)
        transform.SetTranslation(translation_array)

        #create rotation
        slice_center = image.TransformContinuousIndexToPhysicalPoint(
            [(index-1)/2.0 for index in image.GetSize()] )

        angle_z = math.radians(args.angle_z) * sin

        # Rotation matrix around the Z-axis
        rotation_z = [[np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1]]

        # Set Rotation matrix and Center
        transform.SetMatrix(np.ravel(rotation_z))
        transform.SetCenter(slice_center)

        #resample slice using translation transform, setting origin, direction cosine matrix, spacing, size, interpolator, defaultpixelvalue
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputSpacing(image.GetSpacing())
        resampler.SetSize(image.GetSize())
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetDefaultPixelValue(0.0) 
        resampler.SetOutputPixelType(reference.GetPixelID())
        transformed_slice = resampler.Execute(image)

        #take resampled slice and paste into volume copy so that it replaces old, untranslated slice. 
        transformed_image = sitk.Paste(transformed_image, transformed_slice, transformed_slice.GetSize(), destinationIndex=[0, 0, idx])

        sitk.WriteImage(transformed_slice, os.path.join(dir, f'slices_{str(i).zfill(4)}{extension[1]}') ) #write slice to directory

        print(f'Slice Translation {idx}, Translation X {translation_array[0]}, Translation Y {translation_array[1]}')
        print(f'Slice Rotation {idx}, Rotation Z {angle_z}')
        print('\n')

        rot = Rotation.from_euler('xyz', (0, 0, angle_z), degrees=False) #REMEMBER TO CHANGE THE DEGREES FLAG IF NEEDED
        rot_quat = rot.as_quat()

        write_simulated_data(f, args, list(rot_quat[0:3]), list(translation_array), i, slice_center)

    #create reference plots if flag is up
    if args.refplot:
        dirmapping_a = os.getcwd() + ":" + "/data"
        dockerprefix_a = ["docker","run","--rm", "-it", "--init", "-v", dirmapping_a, "--user", str(os.getuid())+":"+str(os.getgid())]

        subprocess.run( dockerprefix_a + ["jauger/motion-monitor:latest", "0000.tfm"])
        # Loop through files in the directory
        for filename in os.listdir(os.getcwd()):
            if filename.endswith(".tfm"):
                os.remove(filename)

    if not os.path.exists('intra_vol'):
        os.makedirs('intra_vol')
    
    sitk.WriteImage(transformed_image, f'intra_vol/intra_vol_{iter}{extension[1]}') #write intra-slice motion volume
    print('Slice Translations Complete')
    f.close()
    

def svr(refVol, slicedir, sms):
    #order files by name
    files = os.listdir(slicedir)
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

    # depending on sms factor, perform svr with slices, looping through until all slices have been registered
    for i in range(0, len(files), sms):
        slicenames = files[i:i + sms]
        slice_list = [os.path.join(slicedir, slicename) for slicename in slicenames]
        outTransFile = str(i).zfill(4)
        
        subprocess.run( dockerprefix + ["crl/sms-mi-reg", "sms-mi-reg", 
        refVol,
        inputTransformFileName,
        outTransFile ] + slice_list )


def write_simulated_data(f, args, rot, trans, i, center):
        #resample slice using translation transform
        transform_array = rot + trans
        a = np.asarray(transform_array).reshape(1,-1)
        np.savetxt(f, a, delimiter=",",fmt="%.8f")
        
        if args.refplot:
            transform1 = sitk.VersorRigid3DTransform()
            transform1.SetCenter(center) #center of rotation
            transform1.SetParameters(transform_array)
            sitk.WriteTransform(transform1, f'./{str(i).zfill(4)}.tfm')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-invol',required=True, help='file path of input volume to perform artificial translation')
    
    parser.add_argument('-x', default = 2, type=float, help='x-axis translation sin wave magnitude')
    parser.add_argument('-y', default = 0, type=float, help='y-axis translation sin wave magnitude')
    parser.add_argument('-angle_z', default = 0, type=float, help='maximum angle of rotation in degrees, z-axis (yaw). Number can be integer or decimal')

    parser.add_argument('-sms_factor', type=int, help='SMS factor. Important for proper motion simulation. Used for fMRI scans') 
    parser.add_argument('--interleaved', action='store_true', help='Flag to add an interleaving factor to change the order of slices')
    parser.add_argument('-interleaved_factor', type=int, help='Interleaved factor. Must be divisible by the number of slices in the volume. Used for HASTE scans')

    parser.add_argument('--svr', action="store_true", help='Flag to perform slice to volume registration')
    parser.add_argument('--refplot', action='store_true', help='create plots of simulated data for a reference')
    parser.add_argument('--slimm', action='store_true', help='flag to perform metadata regeneration on dicom images to use on slimm')

    args = parser.parse_args()

    #create directory if it doesn't exist
    directory = "simulated slices"  
    if not os.path.exists(directory):
        os.makedirs(directory)

    extension = os.path.splitext(args.invol) #file extension name for writing purposes

    for i in range(5):
        apply_translation(args, directory, extension, i) 

    #only do metadata regeneration step if the file is a dicom
    if extension[1] == '.dcm' and args.slimm:
        metadata(args.invol, directory)

    #perform svr and motion monitor if the flag is used
    if args.svr:
        svr(args.invol, directory, args.sms_factor)

        # Motion monitor
        os.remove("identity-centered.tfm")
        dirmapping_a = os.getcwd() + ":" + "/data"
        dockerprefix_a = ["docker","run","--rm", "-it", "--init", "-v", dirmapping_a, "--user", str(os.getuid())+":"+str(os.getgid())]
        subprocess.run( dockerprefix_a + ["jauger/motion-monitor:latest", "sliceTransform0000.tfm"])

        # Delete all tfm files
        for filename in os.listdir(os.getcwd()):
            if filename.endswith(".tfm"):
                os.remove(filename)