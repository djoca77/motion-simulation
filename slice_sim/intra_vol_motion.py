import SimpleITK as sitk
import os
import math
import argparse
import numpy as np
import subprocess
from pydicom import dcmread
from scipy.spatial.transform import Rotation


def write_simulated_data(f, refplot, rot, trans, i, center):
    '''
    Function that writes the 6 simulated motion parameters to a csv file and creates transform files for each slice or set of slices, all to create a reference plot, if the refplot flag is active

    param f: csv file
    param refplot: flag for creating reference plot
    param rot: 3 rotation parameters in versor representation
    param trans: 3 translation parameters
    param i: slice index to write transform file name
    param center: fixed parameters to set center of rotation
    '''
    #write transform
    transform_array = rot + trans
    a = np.asarray(transform_array).reshape(1,-1)
    np.savetxt(f, a, delimiter=",",fmt="%.8f")
    
    if refplot:
        transform1 = sitk.VersorRigid3DTransform()
        transform1.SetCenter(center) #center of rotation
        transform1.SetParameters(transform_array)
        sitk.WriteTransform(transform1, f'./{str(i).zfill(4)}.tfm')

        
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


def interleaved_array(size, interleaved_factor):
    '''
    For HASTE images, in order to simulate acquisition order the interleaving factor can be specified and the slice order will be set based on that

    param size: number of slices
    param interleaved_factor: interleaved factor specified by user

    return interleaved array
    '''
    interleaved_array = []
    current_value = 0
    base_value = 0

    while interleaved_factor > base_value:
        if current_value < size:
            interleaved_array.append(current_value)
            current_value += interleaved_factor #increment by interleaved factor
        else:
            # Set the starting point for the next cycle
            base_value += 1
            current_value = base_value
        print(current_value)

    return interleaved_array


def aquisition_time(inVol):
    '''
    Based on the acquisition times of each slice in the volume, the array returned is the indices ordered by acquisition time from first to last. The sms factor is also ascertained from the acquistions times
    For example, an array of acquisition times like [0.5, 1.5, 2.5, 3.5, 0.5, 1.5, 2.5, 3.5] will lead to an array [0,4,1,5,2,6,3,7] because slices 0 and 4 were acquired first and at the same time, etc. the sms factor
    therefore would be determined to be 2. This is done automatically, so sms_factor does not need to be specified

    param inVol: Input filename for dicom volume

    return sorted array and sms factor
    '''
    dcm = dcmread(inVol)
    uSliceTime = np.empty(len(dcm.PerFrameFunctionalGroupsSequence))
    for iSlice in range(len(dcm.PerFrameFunctionalGroupsSequence)):
        uSliceTime[iSlice] = float(dcm.PerFrameFunctionalGroupsSequence[iSlice].FrameContentSequence[0].FrameAcquisitionDateTime) 

    _, counts = np.unique(uSliceTime, return_counts=True)
    sms = counts.max()
    
    sortedSliceTime = [i for i, _ in sorted(enumerate(uSliceTime), key=lambda x: x[1])]

    return sortedSliceTime, sms


def svr(refVol, slicedir, sms):
    '''
    Perform Slice-to-Volume registration on the input volume and the modified slices to see how well sms-mi-reg does at tracking motion

    param refVol: reference volume that the slices are registered to
    param slicedir: directory of slices with artificial motion applied
    param sms: sms factor that determines how many slices are input into sms-mi-reg at a time
    
    '''
    #order files by name
    files = os.listdir(slicedir)
    files.sort()

    # host computer : container directory alias
    dirmapping = os.getcwd() + ":" + "/data"
    dockerprefix = ["docker","run","--rm", "-it", "--init", "-v", dirmapping,
        "--user", str(os.getuid())+":"+str(os.getgid())]
    print(dockerprefix)

    #create input transform file as starting point for registration
    inputTransformFileName = "identity-centered.tfm"
    subprocess.run( dockerprefix +  [ "crl/sms-mi-reg", "crl-identity-transform-at-volume-center.py", 
    "--refvolume", refVol, 
    "--transformfile", inputTransformFileName ] )

    # depending on sms factor, perform svr with slices, looping through until all slices have been registered
    for i in range(0, len(files), sms):
        slicenames = files[i:i + sms] #take number of slices determined by sms factor
        slice_list = [os.path.join(slicedir, slicename) for slicename in slicenames]

        outTransFile = str(i).zfill(4) #number of transform file
        
        subprocess.run( dockerprefix + ["crl/sms-mi-reg", "sms-mi-reg", 
        refVol,
        inputTransformFileName,
        outTransFile ] + slice_list )


def apply_motion(args, dir, extension):
    '''
    Resample input volume based on specified parameters. Order of slices is determined by the acquisition time for fMRI dicoms or interleaved factor for HASTE images

    param args: args
    param dir: directory for slices
    param extenstion: extension of reference volume
    '''
    try:
        reference = sitk.ReadImage(args.inVol) #read in vol
    except:
        print("Input Image is not Valid") #throw an error otherwise
        exit(1)
    size = reference.GetSize() #returns 3D size array
    num_slices = int(size[2])

    if os.path.exists('parameters.csv'):   
        os.remove('parameters.csv') #remove
    f = open('parameters.csv', 'w') #reopen in write mode

    transformed_image = sitk.Image(reference) #create a copy of volume to modify with slice motion

    if extension == '.dcm':
        #creates array of indices based on the aquisition time of the slices to create a better simulation of the data. Only works for DICOM images since they have the metadata.
        indices, sms = aquisition_time(args.inVol) 
    elif args.interleaved:
        #creates interleaved array depending on sms factor that determines the order of slices
        indices = interleaved_array(num_slices, args.interleaved_factor) 
        sms = args.sms_factor
    else:
        #default is taking slices in order positionally from top to bottom with an sms factor of 1
        indices = list(range(num_slices)) 
        sms = 1
    
    print(f'Order of slices: {indices}')

    for i, idx in enumerate(indices): #for every slice in the volume
        #extract slice
        startIndex = (0, 0, idx)
        sizeROI = (size[0], size[1], 1)
        image = sitk.RegionOfInterest(reference, sizeROI, startIndex)
        
        #create 3D array of sinusoidal motion to be applied translationally to slice
        sin = math.sin(( 2 * math.pi / ( num_slices /  sms )) * ( i // sms)) #FOR SIMULTANEOUS ACQUISITION SIMULATION TAKES THE FLOOR SO IT GOES 0 0 1 1 2 2 3 3 ETC FOR SMS 2
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

        sitk.WriteImage(transformed_slice, os.path.join(dir, f'slices_{str(i).zfill(4)}.nii') ) #write slice to directory

        print(f'Slice Translation {idx}, Translation X {translation_array[0]} mm, Translation Y {translation_array[1]} mm')
        print(f'Slice Rotation {idx}, Rotation Z {math.degrees(angle_z)}')
        print('\n')

        #change rotation parameters from euler to versor since sms-mi-reg is in versor
        rot = Rotation.from_euler('xyz', (0, 0, angle_z), degrees=False) #REMEMBER TO CHANGE THE DEGREES FLAG IF NEEDED
        rot_quat = rot.as_quat()

        write_simulated_data(f, args.refplot, list(rot_quat[0:3]), list(translation_array), i, slice_center)

    #create reference plots if flag is up
    if args.refplot:
        dirmapping_a = os.getcwd() + ":" + "/data"
        dockerprefix_a = ["docker","run","--rm", "-it", "--init", "-v", dirmapping_a, "--user", str(os.getuid())+":"+str(os.getgid())]

        subprocess.run( dockerprefix_a + ["jauger/motion-monitor:latest", "0000.tfm"])
        # Loop through files in the directory
        for filename in os.listdir(os.getcwd()):
            if filename.endswith(".tfm"):
                os.remove(filename)

    #create folder for volume file with
    if not os.path.exists('intra_vol'):
        os.makedirs('intra_vol')
    
    sitk.WriteImage(transformed_image, f'intra_vol/intra_vol{extension}') #write intra-slice motion volume
    print('Slice Translations Complete')
    f.close()


if __name__ == '__main__':
    '''
    This script handles intra-volume motion on a slice by slice basis. Based on the amount of motion desired in the x and y axis translationally and the z axis rotationally, along with
    the sms factor, each slice(s) will be translated independent of the rest of the volume. Each slice is written as its own file, and is always written as a .nii file because .dcm files have issues
    with metadata such as spacing and origin. There are 3 flags to perform slice-to-volume registration, create a reference plot to compare with SVR results, and a slimm flag that adds relevant metadata
    for proper operation in SLIMM (this only is applicable to DICOMS)
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('inVol', help='file path of input volume to perform artificial translation')
    
    parser.add_argument('-x', default = 0, type=float, help='x-axis translation sin wave magnitude')
    parser.add_argument('-y', default = 0, type=float, help='y-axis translation sin wave magnitude')
    parser.add_argument('-angle_z', default = 10, type=float, help='maximum angle of rotation in degrees, z-axis (yaw). Number can be integer or decimal')

    parser.add_argument('-sms_factor', type=int, help='SMS factor. Important for proper motion simulation. Used for fMRI scans') 
    parser.add_argument('-interleaved_factor', type=int, help='Interleaved factor. Must be divisible by the number of slices in the volume. Used for HASTE scans')
    parser.add_argument('--interleaved', action='store_true', help='Flag to add an interleaving factor to change the order of slices')
    
    parser.add_argument('--svr', action="store_true", help='Flag to perform slice to volume registration')
    parser.add_argument('--refplot', action='store_true', help='create plots of simulated data for a reference')
    parser.add_argument('--slimm', action='store_true', help='flag to perform metadata regeneration on dicom images to use on slimm')

    args = parser.parse_args()

    #create directory if it doesn't exist
    directory = "simulated slices"  
    if not os.path.exists(directory):
        os.makedirs(directory)

    extension = os.path.splitext(args.inVol) #file extension name for writing purposes

    apply_motion(args, directory, extension[1]) #run simulated motion

    #only do metadata regeneration step if the file is a dicom. metadata regeneration is done do the whole reconstructed volume
    if extension[1] == '.dcm' and args.slimm:
        metadata(args.inVol, "./intra_vol")

    #perform svr and motion monitor if the flag is used
    if args.svr:
        svr(args.inVol, directory, args.sms_factor)

        # Motion monitor
        os.remove("identity-centered.tfm")
        dirmapping_a = os.getcwd() + ":" + "/data"
        dockerprefix_a = ["docker","run","--rm", "-it", "--init", "-v", dirmapping_a, "--user", str(os.getuid())+":"+str(os.getgid())]
        subprocess.run( dockerprefix_a + ["jauger/motion-monitor:latest", "sliceTransform0000.tfm"])

        # Delete all tfm files
        for filename in os.listdir(os.getcwd()):
            if filename.endswith(".tfm"):
                os.remove(filename)