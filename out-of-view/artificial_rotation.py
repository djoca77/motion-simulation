from pydicom import dcmread
import SimpleITK as sitk
import os
import math
import argparse
import numpy as np
import subprocess

def apply_rotation(args):
    """
    Resample volume, creating a copy of the reference volume rotated in a sinusoidal fashion
    """

    os.remove('rotation.csv')
    f = open('rotation.csv', 'w') #reopen in append mode

    reference = sitk.ReadImage(args.inVol) #read in vol

    #metadata
    reader = sitk.ImageFileReader()
    reader.SetFileName(args.inVol)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    #get file extension for saving purporses
    extension = os.path.splitext(args.inVol)

    for i in range(int(args.num_dicoms)):        
        # Center of volume.
        reference_center = reference.TransformContinuousIndexToPhysicalPoint(
            [(index-1)/2.0 for index in reference.GetSize()] )

        transform = sitk.AffineTransform(3)
        transform.SetCenter(reference_center)
        # Rotation 2D :
        # R = (cos t, -sin t, sin t, cos t)
        # x = 0, y = 1, z = 2
        # 0,1 yaw (z)
        # 0,2 pitch (y)
        # 1,2 roll (x)
        sin = math.sin(0.2*i) 
        angle = args.angle_max * sin
        axis0 = 0 
        axis1 = 1
        theta = np.radians( angle )

        transform.Rotate(axis0, axis1, theta)
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
        
        if args.crop:
            transformed_image = crop(transformed_image)

        sitk.WriteImage(transformed_image, f'./rotated_vols/rotated_{str(i).zfill(4)}{extension[1]}')
        print(f'Volume Rotation {i}, Rotation Matrix: {angle} Degrees')
        
        if args.refplot: write_simulated_data(f, args, theta, i)

    f.close()

def write_simulated_data():
        dirmapping_a = os.getcwd() + ":" + "/data"
        dockerprefix_a = ["docker","run","--rm", "-it", "--init", "-v", dirmapping_a, "--user", str(os.getuid())+":"+str(os.getgid())]

        subprocess.run( dockerprefix_a + ["jauger/motion-monitor:latest", "sliceTransform0000.tfm"])
        # Loop through files in the directory
        for filename in os.listdir(os.getcwd()):
            if filename.endswith(".tfm"):
                os.remove(filename)

def vvr(refVol, voldir):
    """
    Performs volume to volume registration on all simulated volumes, with the unmoved volume being the reference. Output will be .tfm files that 
    """
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

        subprocess.run( dockerprefix + ["crl/sms-mi-reg", "sms-mi-reg", 
        refVol,
        inputTransformFileName,
        outTransFile,
        vol] )

def crop(image):
    low = [25, 15, 0] # X, Y, Z pixels respectively
    up = [25, 8, 0] # X, Y, Z pixels respectively
    cropped = sitk.Crop(image, low, up)

    return cropped

def write_simulated_data(f, args, theta, i):
        #resample slice using translation transform
        transform_array =(0,0,theta,0,0,0)
        a = np.asarray(transform_array).reshape(1,-1)
        np.savetxt(f, a, delimiter=",",fmt="%.8f")
        transform1 = sitk.VersorRigid3DTransform()
        transform1.SetCenter((0,0,0)) #center of rotation
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
    parser.add_argument('-angle_max', default=15, help='maximum angle of rotation in degrees')

    parser.add_argument('--vvr', action='store_true', help='flag for performing volume to volume registration')
    parser.add_argument('--crop', action='store_true', help='flag to all cropping of image or not')
    parser.add_argument('--refplot', action='store_true', help='flag to create motion monitor plots for simulated motion parameters')

    args = parser.parse_args()

    directory = "rotated_vols"
    if not os.path.exists(directory):
        os.makedirs(directory)

    apply_rotation(args)

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

