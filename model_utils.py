import torch

import numpy as np
import pytorch_lightning as pl
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import streamlit as st
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2023, 0.1994, 0.2010)

SHAPE = (224, 224)

CLASS_WEIGHTS = np.array([
    4.37527304,
    2.78349083,  
    1.30183284, 
    12.44099379,  
    1.28545758,
    0.21338021, 
    10.07545272
])

CLASS_WEIGHTS_BINARY = np.array([
    0.56251404, 
    4.49910153
])

transform_1 = transforms.Resize(SHAPE)

transform_2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


class RedNeuronal(pl.LightningModule):
    def __init__(self, class_weights=None):
        super(RedNeuronal, self).__init__()
        if class_weights is None:
            self.class_weights = torch.FloatTensor(CLASS_WEIGHTS)
        else:
            self.class_weights = torch.FloatTensor(class_weights)

        self.conv1 = nn.Conv2d(3, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1   = nn.Linear(16 * 54 * 54, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, len(self.class_weights))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, 
                               weight = self.class_weights.type_as(x))
        self.log("train_loss_step", loss, on_epoch=True)
        self.accuracy(y_hat, y)
        self.log("train_acc_step", self.accuracy, on_epoch=True)
        self.f1(y_hat, y)
        self.log("train_f1_step", self.f1, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, 
                               weight = self.class_weights.type_as(x))
        self.log("val_loss", loss)
        self.accuracy_val(y_hat, y)
        self.log("val_acc_step", self.accuracy_val, on_epoch=True)
        self.f1_val(y_hat, y)
        self.log("val_f1_step", self.f1_val, on_epoch=True)
        return y_hat

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = 1e-5)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def load_model(checkpoint):
    with torch.no_grad():
        model = RedNeuronal.load_from_checkpoint(checkpoint)
        model.to(device)
        model.eval()
        return model

def eval_image(model, image_orig):
    image_resized = transform_1(image_orig)
    tens = transform_2(image_resized)

    tens = tens.unsqueeze(0)
    logits = model(tens)
    probs = F.softmax(logits, dim=1)
    probs = probs.detach().numpy()
    probs = probs[0]

    with GradCAM(model=model, target_layers=[model.conv2], use_cuda=False) as cam:
        grayscale_cam = cam(input_tensor = tens)
        grayscale_cam = grayscale_cam[0, :]
    
    image_arr = np.array(image_resized) / 255
    overlay = show_cam_on_image(image_arr, grayscale_cam, use_rgb=True)
    overlay_img = Image.fromarray(overlay)

    return probs, image_resized, overlay_img
