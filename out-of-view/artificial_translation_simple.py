import SimpleITK as sitk
import os
import argparse

def apply_translation(args):
    image = sitk.ReadImage(args.input_img)

    reader = sitk.ImageFileReader()
    reader.SetFileName(args.input_img)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    
    translation_array = (args.x, args.y, args.z)
    transform = sitk.AffineTransform(3)
    transform.SetTranslation(translation_array)

    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputSpacing(image.GetSpacing())
    resampler.SetSize(image.GetSize())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetOutputPixelType(sitk.sitkUInt16)

    transformed_image = resampler.Execute(image)

    for j in (reader.GetMetaDataKeys()):
        transformed_image.SetMetaData(j, reader.GetMetaData(j))
    
    if not os.path.exists('./translated'):
        os.makedirs('./translated')

    sitk.WriteImage(transformed_image, f'./translated/croppedadultjoshtran.dcm')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-x', default=8, help='x-axis translation sin wave magnitude')
    parser.add_argument('-y', default=0, help='y-axis translation sin wave magnitude')
    parser.add_argument('-z', default=0, help='z-axis translation sin wave magnitude')
    parser.add_argument('-input_img', default='./input/croppedadultjosh.dcm', help='input folder with single dicom file that is duplicated')

    args = parser.parse_args()

    apply_translation(args)
    