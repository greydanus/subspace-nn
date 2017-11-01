# subspace-nn: neural network optimization in fewer dimensions
# Sam Greydanus. June 2017. MIT License.

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, models, transforms, utils
import numpy as np

# data loader
class Dataloader():
    def __init__(self, batch_size, input_dim):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.modes = modes = ['train', 'test']
        trans = transforms.Compose([transforms.ToTensor(),]) # transforms.Normalize((0.1307,), (0.3081,))
        dsets = {k: datasets.MNIST('./data', train=k=='train', download=True, transform=trans) for k in modes}
        self.loaders = {k: torch.utils.data.DataLoader(dsets[k], batch_size=batch_size, shuffle=True) for k in modes}

    def mnist(self, mode='train'):
        X, y = next(iter(self.loaders[mode]))
        return Variable(X).resize(self.batch_size, self.input_dim), Variable(y)

def accuracy(model, dataloader, mode='test'):
    assert mode in dataloader.modes, 'incorrect mode supplied'
    model.eval()
    loss = 0 ; correct = 0
    nbatches = int(10000/model.batch_size)
    for _ in range(nbatches):
        X, y = dataloader.mnist(mode)
        y_hat = model(X)
        loss += F.nll_loss(y_hat, y).data[0]
        pred = y_hat.data.max(1)[1]
        correct += pred.eq(y.data).cpu().sum()

    loss /= model.batch_size*nbatches # loss function already averages over batch size
    total = model.batch_size*nbatches
    model.train()
    return loss, correct, total

def train(dataloader, model, optimizer, global_step=0, total_steps=1000, test_every=100):
    running_loss = None
    acc_msg = '...' ; print('\ttraining...')
    loss_hist = []
    acc_hist = []

    # generic train loop
    for global_step in range(global_step, total_steps+global_step+1):

        # ======== DISCRIMINATOR STEP ======== #
        # forward
        X, y = dataloader.mnist(mode='train')
        y_hat = model(X)

        # backward
        loss = F.nll_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        np_loss = loss.data.numpy()[0]
        running_loss = np_loss if running_loss is None else .99*running_loss + (1-.99)*np_loss
        loss_hist.append((global_step, running_loss))

        # ======== DISPLAY PROGRESS ======== #
        print('\tstep {}/{} | {} | loss: {:.4f}'.format(global_step, total_steps, \
            acc_msg, running_loss), end="\r")
        if global_step % test_every == 0:
            l, c, t = accuracy(model, dataloader, mode='test')
            acc_msg = 'accuracy: {:.4f}% ({}/{})'.format(100*c/t, c, t)
            acc_hist.append((global_step, 100*c/t))

    return loss_hist, acc_hist

