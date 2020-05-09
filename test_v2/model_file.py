import torch
import torch.nn as nn
import torch.nn.functional as F

# The full road model holds and image model (ResNet18), and applies that
# image model to each of the 6 road images.
class RoadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', 
                                          pretrained=False)
        self.fc1 = nn.Linear(6000, 2500)
        self.ups1 = nn.Upsample(size=100, mode='bilinear')
        self.deconv1 = nn.ConvTranspose2d(in_channels=1, out_channels=32, 
                                          kernel_size=3, stride=1, padding=1)
        self.ups2 = nn.Upsample(size=200, mode='bilinear')
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, 
                                          kernel_size=3, stride=1, padding=1)
        self.ups3 = nn.Upsample(size=400, mode='bilinear')
        self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=4, 
                                          kernel_size=3, stride=1, padding=1)
        self.ups4 = nn.Upsample(size=800, mode='bilinear')
        self.deconv4 = nn.ConvTranspose2d(in_channels=4, out_channels=1, 
                                          kernel_size=3, stride=1, padding=1)
    
    
    def forward(self, x):
        features = []
        for im in x:
            features.append(self.image_model(im))
        x = torch.cat(features, dim=1)
        x = self.fc1(x)
        # x = F.relu(x)
        x = x.reshape([-1, 1, 50, 50])
        x = self.ups1(x)
        x = self.deconv1(x)
        x = F.relu(x)
        x = self.ups2(x)
        x = self.deconv2(x)
        x = F.relu(x)
        x = self.ups3(x)
        x = self.deconv3(x)
        x = F.relu(x)
        x = self.ups4(x)
        x = self.deconv4(x)
        return F.sigmoid(x)
    

# Bounding Box Predictor
class BoundingBoxModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    
    def forward(self, x):
        pass