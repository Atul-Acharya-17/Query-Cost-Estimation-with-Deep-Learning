import os
from pathlib import Path
import torch
from dotenv import load_dotenv


load_dotenv()

DATA_ROOT = Path(os.environ["DATA_ROOT"])
PKL_PROTO = 4
DATABASE_URL = os.environ["DATABASE_URL"]

SAMPLE_NUM = 1000

NUM_TRAIN = 1000
NUM_VAL = 100
NUM_TEST = 100

BATCH_SIZE = 64