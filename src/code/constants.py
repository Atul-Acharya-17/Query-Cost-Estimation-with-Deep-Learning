import os
from pathlib import Path
import torch
from dotenv import load_dotenv


load_dotenv()

DATA_ROOT = Path(os.environ["DATA_ROOT"])
PKL_PROTO = 4
DATABASE_URL = os.environ["DATABASE_URL"]

SAMPLE_NUM = 1000

NUM_TRAIN = 10000
NUM_VAL = 1000
NUM_TEST = 1000

JOB_TRAIN = 100000
JOB_LIGHT = 70
SCALE = 500
SYNTHETIC = 5000

BATCH_SIZE = 64