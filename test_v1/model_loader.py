"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import os
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.rcParams['figure.figsize'] = [5, 5] 
matplotlib.rcParams['figure.dpi'] = 200
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from data_helper import UnlabeledDataset, LabeledDataset
from google.colab.patches import cv2_imshow
from helper import collate_fn, draw_box

# import your model class
from model_file import RoadModel


# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'team_name'
    round_number = 1
    team_member = []
    contact_email = '@nyu.edu'

    def __init__(self, model_file='put_your_model_file_name_here'):
        # You should 
        #       1. create the model object
        model = RoadModel()
        #       2. load your state_dict
        model.load_state_dict(torch.load(model_file))
        #       3. call cuda()
        device = torch.device("cuda")
        # self.model = ...
        self.model = model.to(device)

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        
        return torch.rand(1, 800, 800) > 0.5
