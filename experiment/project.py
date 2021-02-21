from seldonian.seldonian import *
from seldonian.synthetic import *
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
import os
from time import time

import argparse
import json
import ray
import copy

import torch

has_gpu = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Process some config values')
parser.add_argument("--config")
parser.add_argument("checkpoint", nargs='?', default='results-' + str(time()))
parser.add_argument("--threads")
parser.add_argument("--dir")
parser.add_argument("--gpus")
parser.add_argument("--workers")
parser.add_argument("--local")

args = parser.parse_args()
if args.local:
    use_local = True
else:
    use_local = False
if args.dir:
    dir = args.dir
else:
    dir = 'results_default'

kwargs = {
    'local_mode': use_local
}
if args.gpus:
    n_gpus = int(args.gpus)
    kwargs['num_gpus'] = n_gpus
else:
    n_gpus = 0

if args.workers:
    workers = int(args.workers)
else:
    workers = 5

if args.config:
    exp_config = json.load(open(args.config, "r"))
else:
    exp_config = {'N': np.geomspace(1e2, 1e6, 20).astype(np.int), 'trials': 40,
                  'methods': ['ttest', 'hoeffdings'], 'D': 10, 'tprs': [0.3, 0.8],
                  'test_size': 0.4, 'opt': 'Powell'}

uc_result = []
results = {}

if args.checkpoint is None:
    checkpoint = str(np.random.randint(1e4))
else:
    checkpoint = args.checkpoint

