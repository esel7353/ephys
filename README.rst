================================================================================
 ephys
================================================================================

Note: This project is still unfinished, not well tested, and especially not well
documented. Probably there are also thousands of typos and grammatical errors to
be spotted.

esel's physics library. A Python module with useful code for doing physics. The
functionality especially covers the requirements of the physical
'Fortgeschrittenen Praktikum' at the University of Freiburg.

Installation
================================================================================

Run the following lines in this directory to install this program.

.. code:: bash

  sudo make install

You might have to install some python packages (or even python) by yourself. The
following command is supposed to install all required  packages on Ubuntu 14.

.. code:: bash

  sudo apt-get install python python-scipy python-matplotlib easy_install \
    python-setuptools ipython python3 python3-scipy python3-matplotlib \
    easy_install3 python3-setuptools ipython

This installs scipy (including numpy), matplotlib, easy_install, setup-tools and
ipython (a very handy interactive version of python) for python 2 and 3 for
compatibility reasons although ephys only supports python 3.

Documentation
================================================================================

The full documentation of all functions and features is supposed to be in the
code. I hope I can will provide a full reference based on the doc strings some
time in the future. Until then use

.. code:: bash

  pydoc3 ephys.MODULE

to get a quick reference. For example

.. code:: bash

  pydoc3 ephys.data.Quantity

prints the class reference of the Quantity class.

Tutorial
================================================================================

The directory tut/ holds some example data, example scripts and a pdf
document, which should introduce the most common functionalities of ephys.
