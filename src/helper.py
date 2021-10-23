
import numpy as np
import random
import os
import torch 


def fix_random_seed(seed_value=2021):
    os.environ['PYTHONPATHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    # torch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deteministic = True
    torch.backends.cudnn.benchmark = True

    return
