#!/bin/bash
python -m numpy.f2py -m _nnls -c nnls.pyf nnls.f
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/gcc/14.2.0_1/lib/gcc/14/
