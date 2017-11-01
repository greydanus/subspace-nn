# subspace-nn: neural network optimization in fewer dimensions
# Sam Greydanus. June 2017. MIT License.

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import datasets, models, transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import os

from utils import *
from subspace_nn import SubspaceNN

reseed = lambda: np.random.seed(seed=123) ; ms = torch.manual_seed(123) # for reproducibility
reseed()

# parse arguments
parser = argparse.ArgumentParser(description='Subspace NN')
parser.add_argument('--fig_dir', type=str, default='figures/', metavar='f', help='directory to save figures')
parser.add_argument('--hidden_dim', type=int, default=200, metavar='h', help='hidden dimension of neural network')
parser.add_argument('--batch_size', type=int, default=256, metavar='b', help='batch size for training (default is 256)')
parser.add_argument('--omega_dims', type=list, default=[3, 10, 30, 100, 300, 1000, None], metavar='w', help='sizes for omega subspaces')
parser.add_argument('--lr', type=float, default=3e-3, metavar='lr', help='learning rate (default is 3e-3)')
parser.add_argument('--test_every', type=int, default=100, metavar='te', help='record test accuracy after this number of steps')
parser.add_argument('--epochs', type=int, default=10, metavar='e', help='number of epochs to train (default is 10)')
args = parser.parse_args()

# book-keeping
os.makedirs(args.fig_dir) if not os.path.exists(args.fig_dir) else None
args.input_dim = 28**2
args.target_dim = 10
global_step = 0
total_steps = int(60000*args.epochs/args.batch_size)
print("using {} epochs".format(args.epochs))

# make dataloader
dataloader = Dataloader(args.batch_size, args.input_dim)

# train models with different omega subspace sizes, saving loss and accuracies for each
model = None
histories = []
for i, omega_dim in enumerate(args.omega_dims):
    reseed()
    model = SubspaceNN(args.batch_size, args.input_dim, args.hidden_dim, args.target_dim, omega_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0)
    h = train(dataloader, model, optimizer, global_step, total_steps, args.test_every)
    histories.append(h)

# plot losses

f2 = plt.figure(figsize=[8,5])
for i, omega_dim in enumerate(args.omega_dims):
    omega_dim = model.theta_dim if omega_dim is None else omega_dim
    loss_hist = histories[i][0]
    xy = np.stack(loss_hist)
    plt.plot(xy[:,0], xy[:,1], linewidth=3.0, label='[$\omega$]={}'.format(omega_dim))

title = "2-layer NN loss on MNIST ([$\Theta$]={})".format(model.theta_dim)
plt.title(title, fontsize=16)
plt.xlabel('train step', fontsize=14) ; plt.setp(plt.gca().axes.get_xticklabels(), fontsize=14)
plt.ylabel('loss', fontsize=14) ; plt.setp(plt.gca().axes.get_yticklabels(), fontsize=14)
plt.ylim([0,2.5])
plt.legend()

f2.savefig('./{}subspace-loss.png'.format(args.fig_dir), bbox_inches='tight')

# plot test accuracies

f3 = plt.figure(figsize=[8,5])
for i, omega_dim in enumerate(args.omega_dims):
    omega_dim = model.theta_dim if omega_dim is None else omega_dim
    acc_hist = histories[i][1]
    xy = np.stack(acc_hist)
    plt.plot(xy[:,0], xy[:,1], linewidth=3.0, label='[$\omega$]={}'.format(omega_dim))

title = "2-layer NN test accuracy on MNIST ([$\Theta$]={})".format(model.theta_dim)
plt.title(title, fontsize=16)
plt.xlabel('train step', fontsize=14) ; plt.setp(plt.gca().axes.get_xticklabels(), fontsize=14)
plt.ylabel('accuracy (%)', fontsize=14) ; plt.setp(plt.gca().axes.get_yticklabels(), fontsize=14)

results_msg = 'epochs: {}\nlearning rate : {}\nbatch size: {}\nmax accuracy: {:.2f}%'\
    .format(args.epochs, args.lr, args.batch_size, acc_hist[-1][-1])
f3.text(0.92, .50, results_msg, ha='left', va='center', fontsize=12)
plt.ylim([0,100])
plt.legend()

f3.savefig('./{}subspace-acc.png'.format(args.fig_dir), bbox_inches='tight')

# display ending message
print('finished running experiment. see ./{}* for loss and accuracy plots'.format(args.fig_dir))