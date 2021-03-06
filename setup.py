# 
# LibHEOM, version 0.5
# Copyright (c) 2019-2020 Tatsushi Ikeda
#
# This library is distributed under BSD 3-Clause License.
# See LINCENSE.txt for licence.
# ------------------------------------------------------------------------

from setuptools import setup
import sys
import os

setup(name='pyheom',
      version='0.5',
      author='Tatsushi IKEDA',
      author_email='ikeda.tatsushi.37u@kyoto-u.jp',
      install_requires=['pylibheom>=0.5', 'numpy', 'scipy'],
      packages=['pyheom'],
      package_dir={'pyheom':'pyheom'},
      zip_safe=False)
