''' Configuration File.
'''

CUDA = 'cuda:0'
MODE = 'test' # chose 'test' or 'val'
TRIALS = 1 # number of repeat experiments

# AL Settings
INIT = 50
ADD = 50
CYCLES = 6
SUBSET = 1000 # random subsampling for unlabeled pool

# Training Settings
EPOCH = 100
BATCH = 8
LR = 0.01
WDECAY = 0.0001

# Params for other AL methods
MARGIN = 1.0 # Param for lloss
LLWEIGHT = 1.0 # Param for lloss
TDWEIGHT = 1.0 # Param for TiDAL

# Params for ATL_Seg
BRI = 1. # brightness
CON = 1. # contrast
SAT = 1. # saturation
HUE = 0. # hue
DIS = 'Cos' # chose 'L1', 'L2' or 'Cos'
LAMBDA = 1 # Loss Weight

VOTE = 4 # number of expert for inference