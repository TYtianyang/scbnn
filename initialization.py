# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 13:03:59 2020

@author: Tianyang
"""

import os
import time
import gc
import math
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import scipy
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
from scipy import interpolate
import bspline
import bspline.splinelab as splinelab
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import BFGS
import h5py
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
os.chdir('C:/Users/Tianyang/Documents/Option Project/source')