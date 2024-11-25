import SimpleITK as sitk
import os
import math
import argparse
import numpy as np
import subprocess
import sys

def interleaved_array(size, sms):
    interleaved_factor = int(size / sms)
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

def apply_translation(args):
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
    f = open('parameters.csv', 'a') #reopen in append mode

    transformed_image = sitk.Image(reference) #create a copy of volume to modify with slice motion

    extension = os.path.splitext(args.invol) #file extension name for writing purposes

    interleaved = interleaved_array(num_slices, args.sms) #creates interleaved array depending on sms factor that determines the order of slices
    print(interleaved)

    for i, idx in enumerate(interleaved): #for every slice in the volume
        #extract slice
        startIndex = (0, 0, idx)
        sizeROI = (size[0], size[1], 1)
        image = sitk.RegionOfInterest(reference, sizeROI, startIndex)

        #create 3D array of sinusoidal motion to be applied translationally to slice
        sin = math.sin(( 2 * math.pi / ( num_slices /  args.sms )) * ( idx )) #// int(args.sms)) FOR SIMULTANEOUS ACQUISITION SIMULATION TAKES THE FLOOR SO IT GOES 0 0 1 1 2 2 3 3 ETC
        translation_array = ((int( args.x ) * sin), (int( args.y ) * sin), 0)
        transform = sitk.AffineTransform(3)
        transform.SetTranslation(translation_array)

        #Writing Sinusoidal Motion into transform files
        transform_array = ( 0, 0, 0, (int( args.x ) * sin), (int( args.y ) * sin), 0) #creating 6 parameter array
        a = np.asarray(transform_array).reshape(1,-1) #reshape into a list
        np.savetxt(f, a, delimiter=",",fmt="%.8f") #save the array as a txt 
        transform1 = sitk.VersorRigid3DTransform() #create VersorRigid3DTransform since that's what smsmireg uses
        transform1.SetCenter((0,0,0)) #center of rotation
        transform1.SetParameters(transform_array) #set the 6 parameters for writing
        #sitk.WriteTransform(transform1, f'./{str(i).zfill(4)}.tfm') #write transform

        #resample slice using translation transform, setting origin, direction cosine matrix, spacing, size, interpolator, defaultpixelvalue
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(transform)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputSpacing(image.GetSpacing())
        resampler.SetSize(image.GetSize())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0) 
        transformed_slice = resampler.Execute(image)

        #take resampled slice and paste into volume copy so that it replaces old, untranslated slice. 
        transformed_image = sitk.Paste(transformed_image, transformed_slice, transformed_slice.GetSize(), destinationIndex=[0, 0, idx])

        sitk.WriteImage(transformed_slice, f'./slices_out/slices_{str(i).zfill(4)}{extension[1]}')  #write slice to directory
        print(f'Slice Translation {i}')

        if args.refplot: write_simulated_data(f, args, sin, i, num_slices)

    if not os.path.exists('intra_vol'):
        os.makedirs('intra_vol')
    
    sitk.WriteImage(transformed_image, f'intra_vol/intra_vol{extension[1]}') #write intra-slice motion volume
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


def write_simulated_data(f, args, sin, i, num_slices):
        #Writing Sinusoidal Motion into transform files
        transform_array = ( 0, 0, 0, (int( args.x ) * sin), (int( args.y ) * sin), 0) #creating 6 parameter array
        a = np.asarray(transform_array).reshape(1,-1) #reshape into a list
        np.savetxt(f, a, delimiter=",",fmt="%.8f") #save the array as a txt 
        transform1 = sitk.VersorRigid3DTransform() #create VersorRigid3DTransform since that's what smsmireg uses
        transform1.SetCenter((0,0,0)) #center of rotation
        transform1.SetParameters(transform_array) #set the 6 parameters for writing
        sitk.WriteTransform(transform1, f'./{str(i).zfill(4)}.tfm') #write transform

        if i == (num_slices - 1):
            dirmapping_a = os.getcwd() + ":" + "/data"
            dockerprefix_a = ["docker","run","--rm", "-it", "--init", "-v", dirmapping_a, "--user", str(os.getuid())+":"+str(os.getgid())]

            subprocess.run( dockerprefix_a + ["jauger/motion-monitor:latest", "0000.tfm"])
            # Loop through files in the directory
            for filename in os.listdir(os.getcwd()):
                if filename.endswith(".tfm"):
                    os.remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-x', required=True, type=float, help='x-axis translation sin wave magnitude')
    parser.add_argument('-y', required=True, type=float, help='y-axis translation sin wave magnitude')
    parser.add_argument('-invol',required=True, help='file path of input volume to perform artificial translation')

    parser.add_argument('-svr', action="store_true", help='Flag to perform slice to volume registration')
    parser.add_argument('-sms', required='-svr' in sys.argv, type=int, help='SMS factor. Important for proper motion simulation. MUST BE AN INTEGER') #only required if svr flag is used
    parser.add_argument('--refplot', action='store_true', help='create plots of simulated data for a reference')

    args = parser.parse_args()

    #create directory if it doesn't exist
    directory = "slices_out"  
    if not os.path.exists('slices_out'):
        os.makedirs('slices_out')

    apply_translation(args)

    #perform svr and motion monitor if the flag is used
    if args.svr:
        svr(args.invol, directory, args.sms)

        # Motion monitor
        os.remove("identity-centered.tfm")
        dirmapping_a = os.getcwd() + ":" + "/data"
        dockerprefix_a = ["docker","run","--rm", "-it", "--init", "-v", dirmapping_a, "--user", str(os.getuid())+":"+str(os.getgid())]
        subprocess.run( dockerprefix_a + ["jauger/motion-monitor:latest", "sliceTransform0000.tfm"])

        # Delete all tfm files
        for filename in os.listdir(os.getcwd()):
            if filename.endswith(".tfm"):
                os.remove(filename)