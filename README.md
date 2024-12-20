This is a motion simulation pipeline that can simulate translation and rotation. The image types that are compatible with these pipelines are DICOM and NIFTI images

# Instructions to get started:
- git clone https://github.com/djoca77/motion-simulation.git
- cd motion-simulation
- conda env create -f environment.yml

# Pipelines
- intra_vol_motion.py: Performs slice level motion on a volume. Option to perform SVR on slices
- pipeline.py: Comprehensive volume level motion simulation with options to perform VVR and SVR on said volumes

- artificial_rotation.py: Performs volume level rotation motion simulation with the option to perform VVR
- artificial_translation.py: Performs volume level translation motion simulation with the option to perform VVR
- inter_vol_artificial_motion: Performs volume level translation and rotation motion simulation with the option to perform VVR
