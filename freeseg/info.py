__version__ = '0.7.3'
__author__ = 'Chenghao Liu'
__date__ = '2022.05' # release date

change_log = \
'''

T2-FLAIR Hyperintensities Segmentation Toolbox

CHANGE LOG

>>> 0.7.2    -  minor changes

>>> 0.7.1    -  added post-processing in predict.py

>>> 0.7.0    -  improved stability when terminating the whole pipeline

>>> 0.6.12   -  add new evaluation methods

>>> 0.6.10   -  improved exception handling

>>> 0.6.8    -  code cleaning & improved exception handling

>>> 0.6.5    -  reproducibility check, big update to antsGroupRegistration 
                and training pipeline

>>> 0.6.2    -  security copy of version 0.6.1

>>> 0.6.1    -  antsGroupRegistration bug fixes
                
                improved progress bar display

>>> 0.6.0    -  now antsGroupRegistration can save deformation fields.

>>> 0.5.12a  -  use different strategies for cerebrum, cerebellum,
                brainstem and cortical structure.

                updates antsGroupRegistration.

                code cleaning & refractoring

                remove sparks after label denoising

                now we can export GIF animation of the segmented lesions

>>> 0.5.10b2 -  apply median smoothing to anomaly scores

>>> 0.5.10b  -  added colormap "grayscale2" in nii_preview.py.
             -  code cleaning & refractoring.
             
>>> 0.5.10a  -  first stable release.

'''
