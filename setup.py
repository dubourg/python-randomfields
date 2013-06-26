import sys
import os
from distutils.core import setup
#from distutils.core import Command


#class test(Command):
    #"""Run only the tests concerning features of OTTemplate.
    #"""
    #description = "Automatically run the core test suite for OTTemplate."
    #user_options = []  # distutils complains if this is not here.
    #tests=['ottemplate/test/MyClass_test.py']

    #def initialize_options(self):  # distutils wants this
        #pass

    #def finalize_options(self):    # this too
        #pass

    #def run(self):
        #for test in self.tests:
            #os.system(sys.executable +' '+ test) 
        
setup(name='randomfields',
      version='randomfields.__version__',
      packages=['randomfields'],
      url = "https://github.com/dubourg/python-randomfields",
      description = ("A Python module that implements tools for the simulation and identification of random fields using \
                     the Karhunen-Loeve expansion representation.")
      )