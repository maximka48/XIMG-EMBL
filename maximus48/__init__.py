# -*- coding: utf-8 -*-




# ========Convention: note the difference from PEP8 for variables!=============
# Naming:
#   * classes MixedUpperCase
#   * varables lowerUpper _or_ lower
#   * functions and methods underscore_separated _or_ lower
# =============================================================================

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from maximus48.SSIM import *
from maximus48.SSIM_sf2 import divide
from maximus48.SSIM_sf import *

from maximus48.var import *
from maximus48.polar import *
from maximus48.polar import polar_image

from maximus48.monochromaticCTF import single_distance_CTF
from maximus48.multiCTF2 import *
from maximus48.sidi_phare import Paganin, MBA, BAC, anka_filter



__module__ = "maximus48"
__author__ = \
    "Maxim Polikarpov (European Molecular Laboratory Lab"
__email__ = "polikarpov.maxim@mail.ru"
__version__ = '0.3.4'
__date__ = "1 Aug 2019"
__license__ = "MIT license"

__all__ = ['im_folder',
           'read_stack',
           'show',
           'imrescale',
           'imrescale_interactive',
           'make_video',
           'wavelen',
           
           'SSIM',
           'single_distance_CTF',
           'multi_distance_CTF',
           'shift_image',
           'shift_image_CTF'
           ]
           








