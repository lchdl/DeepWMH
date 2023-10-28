from __future__ import absolute_import
from .pkginfo import __version__, __date__, package_title

print('')
print('')
print('*%s*' % package_title)

_version = 'Version %s (%s)' % (__version__, __date__)
_tlen = len(package_title)
_lpad = (_tlen - len(_version))//2
_rpad = _tlen - len(_version) - _lpad

if _lpad < 0 : _lpad = 0
if _rpad < 0 : _rpad = 0

print('*%s%s%s*' % (' '*_lpad, _version, ' '*_rpad))

print('')
print('')
print('* Please cite the following paper(s) when using the code:')
print('')
print('[1] Chenghao Liu, Zhizheng Zhuo, Liying Qu, Ying Jin, Tiantian Hua, Jun Xu, '
      'Guirong Tan, Yuna Li, Yunyun Duan, Tingting Wang, Zaiqiang Zhang, Yanling Zhang, '
      'Rui Chen, Pinnan Yu, Peixin Zhang, Yulu Shi, Jianguo Zhang, Decai Tian, '
      'Runzhi Li, Xinghu Zhang, Fudong Shi, Yanli Wang, Jiwei Jiang, Aaron Carass, '
      'Yaou Liu, Chuyang Ye. "DeepWMH: a deep learning tool for accurate white matter '
      'hyperintensity segmentation without requiring manual annotations for training". '
      'Science Bulletin, 2023.')
print('')
print('[2] Chenghao Liu, Xiangzhu Zeng, Kongming Liang, Yizhou Yu, Chuyang Ye. '
      '"Improved Brain Lesion Segmentation with Anatomical Priors from Healthy Subjects". '
      'Medical Image Computing and Computer Assisted Intervention (MICCAI), 2021.')
print('')
print('* If you have any question or found any bug in the code, please open an issue or '
      'create a pull request at "https://github.com/lchdl/DeepWMH".')
print('')
print('')

from . import *

