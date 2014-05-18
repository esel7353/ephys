#!/usr/bin/python3

from setuptools import setup

setup(name='ephys',
      install_requires=['matplotlib', 'scipy', 'numpy', 'pylab',
      'shelve'],
      version='0.1',
      description='some physics',
      author='Frank Sauerburger',
      author_email='frank@sauerburger.com',
      url='sauerburger.com',
      packages=['ephys'],
      package_dir={'ephys': 'src'},
      test_suite="test/data.py",
     )
