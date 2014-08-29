# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import core
from core.parameterization import transformations, priors
import models
import mappings
import inference
import util
import examples
import likelihoods
import testing
from numpy.testing import Tester
from nose.tools import nottest
import kern
import plotting

@nottest
def tests():
    Tester(testing).test(verbose=10)

# import os
# if os.name == 'nt':
#     """
#     Fortran seems to like to intercept keyboard interrupts on windows.
#     This means that when a model is optimizing and the user presses Ctrl-C,
#     the program will crash. Since it's kind of nice to be able to stop
#     the optimization at any time, we define our own handler below.

#     """
#     import win32api
#     import thread

#     def handler(sig, hook=thread.interrupt_main):
#         hook()
#         return 1

#     win32api.SetConsoleCtrlHandler(handler, 1)
