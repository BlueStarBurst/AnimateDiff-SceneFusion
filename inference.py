# this is the huggingface inference script for the animation pipeline
# it takes in a text prompt and generates an animation based on the text

import os
import sys
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

