import argparse
import sys
import tensorflow as tf

# import your network file
from nets import cnn, rnn


"""
Default Configuration
"""
NUM_GPUS = int(4)
TEST_MODE = False


"""
k-Fold DATASET Configuration
"""


DATA_FOLD_PATHS = (
    "/mnt/CrossData_v0312/0",
    "/mnt/CrossData_v0312/1",
    "/mnt/CrossData_v0312/2",
    "/mnt/CrossData_v0312/3",
    "/mnt/CrossData_v0312/4",
)
EVAL_FOLD_INDEX       = int(4)

DATA_FOLD_NUM=(
    71795,71808,71763,72023,72004
)
"""
Evaluation Setting
"""
EVAL_SAVE_FIGURES     = False
EVAL_SAVE_FOLDER      = "./results"


# load your own model
MODEL_NETWORK     = cnn.RCNN_BaseLine()
MODEL_SAVE_FOLDER = "/data/"
MODEL_SAVE_NAME   = "TrainSave"
MODEL_OPTIMIZER   = tf.train.AdamOptimizer
MODEL_SAVE_INTERVAL = int(2235)



BATCH_SIZE      = int(256)
IMAGE_WIDTH     = int(128)
IMAGE_HEIGHT    = int(128)
IMAGE_CHANNELS  = int(1)
IMAGE_FRAMES    = int(20)

TRAIN_OVERWRITE = False

TRAIN_DATA_SHUFFLE  = True
TRAIN_MAX_STEPS     = int(200000)
TRAIN_LEARNING_RATE = float(0.0001)
TRAIN_DECAY         = float(0.1)
TRAIN_WEIGHT_DECAY = float(0.00005)
TRAIN_DECAY_INTERVAL= float(1050000)





"""
DO NOT MODIFY BELOW CODE 
"""
EVAL_FOLD_PATHS      = [
    DATA_FOLD_PATHS[EVAL_FOLD_INDEX]
]
TRAIN_FOLD_PATHS     = [ path for path in DATA_FOLD_PATHS if path not in EVAL_FOLD_PATHS ]


"""
Define arguments that is use in every section
"""
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
parser.add_argument('--eval_fold', type=int, default=EVAL_FOLD_INDEX)

# Arguments for training
parser.add_argument('--model_name', type=str, default=MODEL_SAVE_NAME,
                    help='Folder name for saving your trained model.')
parser.add_argument('--overwrite', action='store_true')

# arguments for evaluation
parser.add_argument('--save_figures', action='store_true')


"""
Parsing arguments and Overwrite your default configuration
"""
args = parser.parse_args()

if args.test:
    TEST_MODE = True
if args.overwrite:
    TRAIN_OVERWRITE = True
if args.save_figures:
    EVAL_SAVE_FIGURES = False
if args.model_name:
    MODEL_SAVE_NAME = args.model_name
if args.eval_fold:
    EVAL_FOLD_INDEX = args.eval_fold

