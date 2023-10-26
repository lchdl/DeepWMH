from __future__ import absolute_import
from .info import __version__, __date__

print('')
print('Please cite the following paper when using the code:')
print('')
print('Chenghao Liu, Xiangzhu Zeng, Kongming Liang, Yizhou Yu, Chuyang Ye. '
      '"Improved Brain Lesion Segmentation with Anatomical Priors from Healthy Subjects". '
      'Medical Image Computing and Computer Assisted Intervention (MICCAI), 2021.')
print('')
print('If you have any question or found any bug in the code, please open an issue or '
      'a pull request at https://github.com/lchdl/annotation_free_wmh_seg.')
print('')

print('')
print('* T2-FLAIR Hyperintense Abnormalities *')
print('*        Segmentation Toolbox         *')

_s = 'Version %s (%s)' % (__version__, __date__)
_t = len('T2-FLAIR Hyperintense Abnormalities')
_l = (_t - len(_s))//2
_r = _t - len(_s) - _l

if _l < 0 : _l = 0
if _r < 0 : _r = 0

print('* %s%s%s *' % (' '*_l, _s, ' '*_r))
print('')
print('')

from . import *

