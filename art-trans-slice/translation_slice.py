import SimpleITK as sitk
import os
import math
import argparse
import numpy as np
import re
import subprocess
import sys

def apply_translation(args, extension):
    '''
    Resample volume slice by slice in a sinusoidal fashion 
    '''

    try:
        reference = sitk.ReadImage(args.inVol) #read in vol
    except:
        print("Input Image is not Valid") #throw an error otherwise
        exit(1)
    
    size = reference.GetSize() #returns 3D size array

    os.remove('translation.csv') #remove
    f = open('translation.csv', 'w') #reopen in append mode

    transformed_image = sitk.Image(reference) #create a copy of volume to modify with slice motion

    for i in range(size[2]): #for every slice in the volume
        #extract slice
        startIndex = (0, 0, i)
        sizeROI = (size[0], size[1], 1)
        image = sitk.RegionOfInterest(reference, sizeROI, startIndex)

        #create 3D array of sinusoidal motion to be applied translationally to slice
        sin = math.sin((2*math.pi/size[2])*i)
        translation_array = ((int(args.x) * sin), (int(args.y) * sin), 0)
        transform = sitk.AffineTransform(3)
        transform.SetTranslation(translation_array)

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
        transformed_image = sitk.Paste(transformed_image, transformed_slice, transformed_slice.GetSize(), destinationIndex=[0, 0, i])

        sitk.WriteImage(transformed_slice, f'./slices_out/slices_{str(i).zfill(4)}{extension[1]}')  #write slice to directory
        print(f'Slice Translation {i}')

        if args.refplot: write_simulated_data(f, args, sin, i, size[2])
    
    sitk.WriteImage(transformed_image, f'./intra_vol/intra_vol{extension[1]}') #write intra-slice motion volume
    print('Slice Translations Complete')
    f.close()

def svr(refVol, slicedir, sms):

    # list all slice files and sort them in order
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
    for i, slicename in enumerate(files):
        slice = os.path.join(slicedir, slicename)
        outTransFile = str(i).zfill(4)

        slice_list = []
        if sms == 1:
            slice_list.append(slice)
        elif sms == 2:
            slice_list.append(slice)
            slice = os.path.join(slicedir, files[i + 20])
            slice_list.append(slice)
        elif sms == 4:
            slice_list.append(slice)
            slice = os.path.join(slicedir, files[i + 10])
            slice_list.append(slice)
            slice = os.path.join(slicedir, files[i + 20])
            slice_list.append(slice)
            slice = os.path.join(slicedir, files[i + 30])
            slice_list.append(slice)
        print(slice_list)
        
        subprocess.run( dockerprefix + ["crl/sms-mi-reg", "sms-mi-reg", 
        refVol,
        inputTransformFileName,
        outTransFile ] + slice_list )

        
def write_simulated_data(f, args, sin, i, size):
        #Writing Sinusoidal Motion into transform files
        transform_array =(0,0,0,(int(args.x) * sin), (int(args.y) * sin), 0) #creating 6 parameter array
        a = np.asarray(transform_array).reshape(1,-1) #reshape into a list
        np.savetxt(f, a, delimiter=",",fmt="%.8f") #save the array as a txt 
        transform1 = sitk.VersorRigid3DTransform() #create VersorRigid3DTransform since that's what smsmireg uses
        transform1.SetCenter((0,0,0)) #center of rotation
        transform1.SetParameters(transform_array) #set the 6 parameters for writing
        sitk.WriteTransform(transform1, f'./{str(i).zfill(4)}.tfm') #write transform

        if i == (size - 1):
            dirmapping_a = os.getcwd() + ":" + "/data"
            dockerprefix_a = ["docker","run","--rm", "-it", "--init", "-v", dirmapping_a, "--user", str(os.getuid())+":"+str(os.getgid())]

            subprocess.run( dockerprefix_a + ["jauger/motion-monitor:latest", "0000.tfm"])
            # Loop through files in the directory
            for filename in os.listdir(os.getcwd()):
                if filename.endswith(".tfm"):
                    os.remove(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-x', required=True, help='x-axis translation sin wave magnitude')
    parser.add_argument('-y', required=True, help='y-axis translation sin wave magnitude')
    parser.add_argument('-inVol',required=True, help='file path of input volume to perform artificial translation')
    
    parser.add_argument('--svr', action="store_true", help='Flag to perform slice to volume registration')
    parser.add_argument('sms', required='--svr' in sys.argv, help='SMS factor. Important for proper motion simulation') #only required if svr flag is used
    parser.add_argument('--refplot', action='store_true', help='flag to create plots of simulated data')
    
    args = parser.parse_args()

    directory = "slices_out"  # Replace with your directory path
    if not os.path.exists(directory):
        os.makedirs(directory)

    # extract file extension and file name for proper file writing
    extension = os.path.splitext(args.inVol)

    apply_translation(args, extension) #translation

    #perform svr and motion monitor if the flag is used
    if args.svr:
        svr(args.inVol, directory, int(args.sms))
    
        # Motion monitor
        os.remove("identity-centered.tfm")
        dirmapping_a = os.getcwd() + ":" + "/data"
        dockerprefix_a = ["docker","run","--rm", "-it", "--init", "-v", dirmapping_a, "--user", str(os.getuid())+":"+str(os.getgid())]

        subprocess.run( dockerprefix_a + ["jauger/motion-monitor:latest", "sliceTransform0000.tfm"])
        # Loop through files in the directory
        for filename in os.listdir(os.getcwd()):
            if filename.endswith(".tfm"):
                os.remove(filename)

    