# Parser and directory related
import argparse
import os
from os.path import exists
from rospkg import RosPack

# NN related
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Other scripts in the directory
import neural_network as neural_net
from dataset import ImitationDataset


#torch.manual_seed(10)


class ImitationNet(nn.Module):
    '''
    This class defines the neural network model for supervised learning.
    The inputs are robot pose and laser scans and it outputs robot velocities.
    '''
    def __init__(self, control_dim=2, device='cpu'):
        super(ImitationNet, self).__init__()
        self.device = device
        self.pose_to_control = neural_net.PoseToControl(control_dim=control_dim)

    def forward(self, odom_input, laser_scan):
        control_predict = self.pose_to_control(odom_input, laser_scan)
        return control_predict

    def step(self, odom_input, laser_scan, target_control):
        control_predict = self.forward(odom_input, laser_scan)
        # Squared error loss and sum loss for both linear and angular
        total_loss = ((control_predict-target_control).pow(2)).sum()
        return total_loss

    def save(self, path):
        checkpoint = {'model': self.state_dict()}
        torch.save(checkpoint, path)

    def load(self, path):
        path = os.path.abspath(path)
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])


def train(model, mode='dagger'):
    if mode == 'dagger':
        args.epochs = 50
    elif mode == 'supervised':
        args.epochs = 150
    args.save_dir = os.getcwd() + '/models/' + mode + '/'

    # Load data using dataset script
    train_data = ImitationDataset(device=args.device, mode=mode)

    # Set network to start training
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_decay)
    losses = []
    model.train()

    # Iterate through epochs
    for epoch in range(args.epochs):
        # Reshuffle train and test sets for every epoch
        train_loader = torch.utils.data.DataLoader(train_data, 
                                batch_size=args.batch_size, shuffle=True)

        # Iterations
        for pose, scan, vel in train_loader:
            # Forward and Backward pass
            opt.zero_grad()
            loss = model.step(odom_input=pose, laser_scan=scan, 
                                    target_control=vel)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Print the current status
        print("-" * 25)
        print("Epoch:{:10}".format(epoch))
        print("Train Loss:{:10.6}\t".format(np.mean(losses)))

    # Save and update the model after every full training round
    model.save(args.save_dir + "model" + ".pt")

    return model


def test(model, odom_input, laser_scan):
    odom_input = torch.tensor(odom_input).detach().to(args.device).type(torch.float32)
    laser_scan[laser_scan == np.inf] = 0
    laser_scan = laser_scan[:,::30]
    laser_scan = torch.tensor(laser_scan).detach().to(args.device).type(torch.float32)

    vel_predict = model(odom_input=odom_input, laser_scan=laser_scan)

    return vel_predict.cpu().detach().numpy()


# Organizing all network hyperparameters into a parser upon initalization
os.chdir(RosPack().get_path('cs6244'))
parser = argparse.ArgumentParser(description="network hyperparameters")
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--lr_decay', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--save_dir', type=str, 
                    default=os.getcwd() + '/models/')
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

# configure cuda
if torch.cuda.is_available():
    print('cuda is available')
    args.device = torch.device('cuda')
else:
    print('cuda is not available')
    args.device = "cpu"



# Unit test
def unitTest():
    train(model)
    print("Neural network training round successful")

# Execute the main method
if __name__ == '__main__':
    model = ImitationNet(control_dim=2, device=args.device)
    unitTest()