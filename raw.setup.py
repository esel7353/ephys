#!/usr/bin/python3

from setuptools import setup

setup(name='ephys',
      install_requires=['matplotlib', 'scipy', 'numpy'],
      version='VERSION',
      description='some physics',
      author='Frank Sauerburger',
      author_email='frank@sauerburger.com',
      url='sauerburger.com',
      packages=['ephys'],
      package_dir={'ephys': 'src'},
			scripts=['scripts/ephysdir'],
      #test_suite="test/data.py",
     )
