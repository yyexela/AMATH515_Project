###############################
# Imports # Imports # Imports #
###############################

import torch
from torch import nn

#############################
# CIFAR10 CNN # CIFAR10 CNN #
#############################

class CIFAR10_CNN(nn.Module):
    def __init__(self):
        # See https://realpython.com/python-super/ for info about super
        super(CIFAR10_CNN, self).__init__()
        # CNN part
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=6,kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=6,out_channels=12,kernel_size=3,padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        # Linear part
        self.classifier = nn.Sequential(
            nn.Linear(12*8*8, int(12*8*8/2)),
            nn.ReLU(inplace=True),
            nn.Linear(int(12*8*8/2), int(12*8*8/4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(12*8*8/4), 10),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
        
    def get_label(self, logits):
        pred_prob = nn.Softmax(dim=1)(logits)
        y_pred = pred_prob.argmax(1)
        return y_pred

def save_CIFAR10_CNN(epoch, model, optimizer,\
                     train_acc, test_acc, train_loss, test_loss,\
                     save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_loss': train_loss,
        'train_loss': train_loss,
        }, save_path)
