from numpy.core.function_base import geomspace
from ray.tune import analysis
from actor_critic import *

import ray
from ray import tune
import json
import argparse

from time import time

from numba import typed, typeof, njit, int64, types
from numba.experimental import jitclass
from numba.extending import overload_method, overload


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("worker-id", help="Add worker is here")
    parser.add_argument("out_dir", help="Directory to output results")
    parser.add_argument("--num-training", default=1000 )
    parser.add_argument("--num-test", default=10000)
    parser.add_argument("--j", default=-8.0)
    parser.add_argument("--alpha-actor", default=0.02)
    parser.add_argument("--alpha-critic", default=0.02)
    parser.add_argument("--lambda", default=0.02)
    parser.add_argument("--num-safety", default=0.2)
    return parser.parse_args()

def main():
    
    pass


if __name__ == "__main__":
    main()

