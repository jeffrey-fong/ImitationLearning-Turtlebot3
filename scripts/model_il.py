# Parser and directory related
import argparse
import os
from os.path import exists

# NN related
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Other scripts in the directory
import neural_network as neural_net
from dataset import ImitationDataset


torch.manual_seed(20)


class ImitationNet(nn.Module):
    '''
    This class defines the neural network model for supervised learning.
    The inputs are robot pose (from odom) and laser scans and it outputs
    robot velocities.
    '''
    def __init__(self, control_dim=2, device='cpu'):
        super(ImitationNet, self).__init__()
        self.device = device
        self.pose_to_control = neural_net.PoseToControl(control_dim=control_dim)

    def forward(self, odom_input, batch_size):
        '''
        Forward pass of the model
        '''
        control_predict = self.pose_to_control(odom_input, batch_size)
        return control_predict

    def step(self, odom_input, target_control):
        """
        Perform one forward and backward pass of the network
        """
        control_predict = self.forward(odom_input, odom_input.size(0))
        total_loss = ((control_predict-target_control).pow(2)).mean(dim=0).sum()
        return total_loss

    def save(self, path):
        """
        Save the model to a given path.
        """
        checkpoint = {'model': self.state_dict()}
        torch.save(checkpoint, path)

    def load(self, path):
        """
        Load the model from a given path.
        """
        path = os.path.abspath(path)
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])


def validate(model, test_data, args):
    """
    # Validation on test_data
    """
    # Switch to evaluation mode
    model.eval()

    # Get the predictions and compute validation loss
    for pose, scan, vel in test_data:
        control_predict = model(odom_input=pose, batch_size=pose.size(0))
        loss_val = (control_predict - vel).pow(2).mean().sqrt()

    # Switch back to training mode
    model.train()

    return loss_val


def train(model):
    # Load data using dataset script
    train_data = ImitationDataset(device=args.device)
    test_data = ImitationDataset(is_test=True, device=args.device)

    # model initialization
    model = ImitationNet(control_dim=2, device=args.device)
    # Load any existing model
    if os.path.exists(args.save_dir + 'model.pt'):
        model.load(args.save_dir + 'model.pt')
    model = model.to(args.device)

    # Set network to start training
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lr_decay)
    losses = []
    model.train()

    # Iterate through epochs
    for epoch in range(args.epochs):
        # Reshuffle train and test sets for every epoch
        train_loader = torch.utils.data.DataLoader(train_data, 
                                batch_size=args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, 
                                batch_size=100*args.batch_size, shuffle=True)

        # Iterations
        for pose, scan, vel in train_loader:
            # Forward and Backward pass
            opt.zero_grad()
            loss = model.step(odom_input=pose, target_control=vel)
            loss.backward()
            opt.step()
            # Append to the list of losses
            losses.append(loss.item())

        # Validate every epoch
        loss_val = validate(model=model, test_data=test_loader, args=args)

        # Print the current status
        print("-" * 25)
        print("Epoch:{:10}".format(epoch))
        print("Train Loss:{:10.6}\t".format(np.mean(losses)))
        print("Val Loss:  {:10.6f}".format(loss_val.item()))

    # Save and update the model after every full training round
    model.save(args.save_dir + "model" + ".pt")


def test(model, odom_input):
    # model initialization
    model = ImitationNet(control_dim=2, device=args.device)
    # Load any existing model
    if os.path.exists(args.save_dir + 'model.pt'):
        model.load(args.save_dir + 'model.pt')
    model = model.to(args.device)

    odom_input = torch.tensor(odom_input).detach().to(args.device).type(torch.float32)

    # Normalize input
    odom_input[:,0] = odom_input[:,0] / 7.5
    odom_input[:,1] = odom_input[:,1] / 5

    vel_predict = model(odom_input=odom_input, batch_size=1)

    vel_predict[:,0] = vel_predict[:,0] * 0.5
    vel_predict[:,1] = vel_predict[:,1] * 2.84

    return vel_predict.cpu().detach().numpy()


# Organizing all network hyperparameters into a parser upon initalization
parser = argparse.ArgumentParser(description="network hyperparameters")
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--lr_decay', type=float, default=0.998)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--save_dir', type=str, 
                    default='/home/jeffrey/catkin_ws/src/cs6244/models/')
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
    train(args)
    print("Neural network training round successful")

# Execute the main method
if __name__ == '__main__':
    unitTest()