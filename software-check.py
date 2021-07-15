import re
import math
import time
import glob
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing
import h5py
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using decive "+str(device))
print("All tests passed!")
