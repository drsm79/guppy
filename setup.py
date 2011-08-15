#!/usr/bin/env python
# encoding: utf-8

from distutils.core import setup, Command
from distutils.util import get_platform
import os
import sys

here = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), sys.argv[0])))

class EnvCommand(Command):
   description = "Configure the PYTHONPATH, DATABASE and PATH variables to" +\
   "some sensible defaults, if not already set. Call with -q when eval-ing," +\
   """ e.g.:
       eval `python setup.py -q env`
   """

   user_options = [ ]

   def initialize_options(self):
       pass

   def finalize_options(self):
       pass

   def run(self):
       tests = here + '/test/python'
       source = here + '/src/python'
       # Stuff we want on the path
       exepth = [source + '/WMCore/WebTools',
                 here + '/bin']

       pypath=os.getenv('PYTHONPATH', '').strip(':').split(':')

       for pth in [tests, source]:
           if pth not in pypath:
               pypath.append(pth)

       # We might want to add other executables to PATH
       expath=os.getenv('PATH', '').split(':')
       for pth in exepth:
           if pth not in expath:
               expath.append(pth)

       print 'export PYTHONPATH=%s' % ':'.join(pypath)
       print 'export PATH=%s' % ':'.join(expath)



setup (name = here.split('/')[-1],
      version = '1.0',
      maintainer_email = 'hn-cms-wmDevelopment@cern.ch',
      cmdclass = {
                  #'test' : TestCommand,
                  'env': EnvCommand,
      },
      # base directory for all our packages
      package_dir = {'': 'src/python/'},
      packages = 'guppy',
      data_files = 'src/c/')
