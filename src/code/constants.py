import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

DATA_ROOT = Path(os.environ["DATA_ROOT"])
PKL_PROTO = 4
DATABASE_URL = os.environ["DATABASE_URL"]

RESULT_ROOT = Path(os.environ["RESULT_ROOT"])

SAMPLE_NUM = 1000

NUM_TRAIN = 10000
NUM_VAL = 1000
NUM_TEST = 1000

JOB_TRAIN = 100000
JOB_LIGHT = 70
SCALE = 500
SYNTHETIC = 500
SYNTHETIC_500 = 500

BATCH_SIZE = 64