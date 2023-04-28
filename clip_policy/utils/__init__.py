from .constants import *
from .word_extractor import *
import torch
import numpy as np
import random
from datetime import timedelta

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)