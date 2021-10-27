import torch

# device on which to train
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# train size
VAL_SIZE = 0.2

# embedding sizes
EMB_DIMS = 5

# number of epochs for Training:
N_EPOCHS = 24

# start learning rate
START_LR = 0.001

# start learning rate
END_LR = 0.001

# start learning rate
MAX_LR = 0.01