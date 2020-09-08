import argparse
import time
import os
from os.path import exists

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import neural_network as neural_net
from dataset import ImitationDataset
import matplotlib.pyplot as plt


torch.manual_seed(20)


class ImitationNet(nn.Module):
    def __init__(self, control_dim=2, device='gpu'):
        super(ImitationNet, self).__init__()
        self.pose_to_control = neural_net.PoseToControl(control_dim=control_dim)
        self.device = device

    def forward(self, odom_input, batch_size):
        control_predict = self.pose_to_control(odom_input, batch_size)
        return control_predict

    def step(self, odom_input, target_control):
        batch_size = rgb_image.size(0)
        rgb_image = torch.reshape(rgb_image, 
                            (rgb_image.size(0)*rgb_image.size(1),3,48,64))

        control_predict = self.forward(rgb_image, h0, c0, batch_size, seq_len)
        likelihood_control = (10000 * (ctrl_predict - target_control).pow(2)).mean(dim=0).sum()
        return likelihood_control

    def save(self, path):
        """
        Save the neural model to a given path.
        """
        checkpoint = {'model': self.state_dict()}
        torch.save(checkpoint, path)

    def load(self, path):
        """
        Load the neural model with a given path.
        """
        path = os.path.abspath(path)
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])


def visualize(model, test_data, args):
    """
    # Randomly show several results in the test set, not a formal evaluation
    """
    model.eval()  # Test/evaluate mode

    # Get the predictions
    inputs, inputs_m, results, loss, mse = [], [], [], [], []
    control = []
    # for noisy_data, clean_data in self.test_data:
    for rgb_image, control in test_data:
        h0 = torch.rand(rgb_image.size(0), 1).view(1,rgb_image.size(0),1).to(args.device)
        c0 = torch.rand(rgb_image.size(0), 1).view(1,rgb_image.size(0),1).to(args.device)

        batch_size = rgb_image.size(0)
        seq_len = rgb_image.size(1)
        rgb_image = torch.reshape(rgb_image, \
                (rgb_image.size(0)*rgb_image.size(1),3,48,64))

        ctrl_predict = model(rgb_image=rgb_image, h0=h0, c0=c0, \
                        batch_size=batch_size, seq_len=seq_len)
        mse_u = (ctrl_predict - control).pow(2).mean().sqrt()
        print("test_mse:", mse_u.item())
        '''plt.close()
        plt.plot(control.cpu().detach().numpy()[0:100, 0], color='black')
        plt.plot(ctrl_predict.cpu().detach().numpy()[0:100, 0], color='red')
        plt.draw()
        plt.pause(0.1)'''

        break
    model.train()
    return mse_u


def train(model):
    # Load data using dataset script
    train_data = ImitationDataset(device=args.device)
    test_data = ImitationDataset(is_test=True, device=args.device)
    print(train_data.poses.shape[0], test_data.poses.shape[0])

    # model
    model = ImitationNet(control_dim=1, device=args.device)
    '''if os.path.exists('/home/jeffrey/catkin_ws/src/cs6244/models/current.pt'):
        print('Loading existing model')
        model.load('./models/current.pt')

    path = os.path.join(save_dir, "best_loss" + ".pt")
    path = os.path.join(save_dir, "epoch" + str(950) + ".pt")

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lr_decay)
    # training
    loss_list = list()

    min_loss = 1000.0
    min_test_mse = 10
    min_train_mse = 10
    for epoch in range(epochs):
        train_data_loader = torch.utils.data.DataLoader(train_data, 
                                        batch_size=batch_size, shuffle=True)
        test_data_loader = torch.utils.data.DataLoader(test_data, 
                                        batch_size=100*args.batch_size, shuffle=True)

        # load data, step, backward
        model.train()
        count = 0
        for rgb_image, control in train_data_loader:

            optimizer.zero_grad()
            loss = model.step(rgb_image=rgb_image, control=control)
            loss.backward()

            optimizer.step()
            loss_list.append(loss.item())

        if epoch % 50 is 0:
            # save model
            path = os.path.join(args.save_dir, "epoch" + str(epoch) + ".pt")
            model.save(path)
            path = os.path.join(args.save_dir, "current" + ".pt")
            model.save(path)
            mse_u_test = visualize(model=model, test_data=train_data_loader, args=args)

        # display info
        print("-" * 5)
        print("Epoch:{:10}\t Loss:{:10.6}\t"
              .format(epoch, np.mean(loss_list)))'''


# setup, training, and evaluation
def main(model):
    """
    1. data
    2. model
    3. optimizer
    4. training ...
    :param args:
    :return:
    """
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--Epochs', type=int, default=20)
    parser.add_argument('--lrd', type=float, default=0.998)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--save_dir', type=str, default='./rgb_image_model') # multimodal_save,baseline_camera_save
    parser.add_argument('--lr', type=float, default=0.002)

    args = parser.parse_args()

    # configure
    if torch.cuda.is_available():
        print('cuda is available')
        args.device = torch.device('cuda')
    else:
        print('cuda is not available')
        args.device = "cpu"

    # Load data for specified modalities
    train_data = DuckieTownDataset(device=args.device)
    test_data = DuckieTownDataset(is_test=True, device=args.device)

    # model
    model = DeepNet(control_dim=1, device='gpu')
    if os.path.exists('/home/jeffrey/gym-duckietown/rgb_image_model/current.pt'):
        print('Loading existing model')
        model.load('./rgb_image_model/current.pt')

    path = os.path.join(args.save_dir, "best_loss" + ".pt")
    path = os.path.join(args.save_dir, "epoch" + str(950) + ".pt")

    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.lrd)
    # training
    loss_list = list()

    min_loss = 1000.0
    min_test_mse = 10
    min_train_mse = 10
    for epoch in range(args.Epochs):
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=100*args.batch_size, shuffle=True)

        # load data, step, backward
        model.train()
        count = 0
        for rgb_image, control in train_data_loader:

            optimizer.zero_grad()
            loss = model.step(rgb_image=rgb_image, control=control)
            loss.backward()

            optimizer.step()
            loss_list.append(loss.item())

        if epoch % 50 is 0:
            # save model
            path = os.path.join(args.save_dir, "epoch" + str(epoch) + ".pt")
            model.save(path)
            path = os.path.join(args.save_dir, "current" + ".pt")
            model.save(path)
            mse_u_test = visualize(model=model, test_data=train_data_loader, args=args)

        # display info
        print("-" * 5)
        print("Epoch:{:10}\t Loss:{:10.6}\t"
              .format(epoch, np.mean(loss_list)))


# Organizing all network hyperparameters into a parser upon initalization
parser = argparse.ArgumentParser(description="network hyperparameters")
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--lr_decay', type=float, default=0.998)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--save_dir', type=str, default='./models')
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

# configure cuda
if torch.cuda.is_available():
    print('cuda is available')
    args.device = torch.device('cuda')
else:
    print('cuda is not available')
    args.device = "cpu"



# Execute the main method
if __name__ == '__main__':
    train(args)