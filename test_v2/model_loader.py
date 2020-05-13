"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
from model_file import RoadModel, BoundingBoxModel

# Set up your device 
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'Mini-Batch'
    team_number = 58
    round_number = 3
    team_member = ['Henry Steinitz', 'Chady Raach', 'Jatin Khilnani']
    contact_email = 'jk6373@nyu.edu'

    def __init__(self, model_file='model_state_dict.pt'):
        # You should 
        #       1. create the model object
        ri_model = RoadModel()
        bb_model = BoundingBoxModel()
        #       2. load your state_dict
        model_state_dict = torch.load(model_file)
        ri_model.load_state_dict(model_state_dict['ri_model'])
        bb_model.load_state_dict(model_state_dict['bb_model'])
        #       3. call cuda()
        self.ri_model = ri_model.to(device)
        self.bb_model = bb_model.to(device)
        # 
        # pass

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        
        # return (torch.rand(1, 15, 2, 4) * 80) - 40

        self.bb_model.eval()
        return self.bb_model(samples.permute(1, 0, 2, 3, 4))[:,:,:8].reshape([-1,1,2,4])

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 
        
        # return torch.rand(1, 800, 800) > 0.5
        
        self.ri_model.eval()
        return self.ri_model(samples.permute(1,0,2,3,4)) > 0.5
