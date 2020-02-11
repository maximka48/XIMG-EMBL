# -*- coding: utf-8 -*-
from setuptools import setup


long_description = u"""

Official Github page of the project - https://github.com/maximka48/XIMG-EMBL.

This package is dedicated to image processing, flatifield correction, phase retrieval and X-ray tomography. You may find some examples at GitHub however you may also contact me (Maxim Polikarpov, polikarpov.maxim@mail.ru) in case you would like to use this package and get some guidance. In addition, each function or class have a situational description.

MIT License, Copyright (c) 2019 Maxim Polikarpov
"""


setup(
    name='maximus48',
    version='1.1.3',
    description='Useful tools for image processing & parallel-beam X-ray tomography',
    keywords=['tomography', 'reconstruction', 'imaging'],
    long_description=long_description,
    author='Maxim Polikarpov',
    author_email='polikarpov.maxim@mail.ru',
    url='https://github.com/maximka48/maximus48',
    platforms='OS Independent',
    license='MIT License',
    packages=['maximus48'])
