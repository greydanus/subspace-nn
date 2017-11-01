# subspace-nn: neural network optimization in fewer dimensions
# Sam Greydanus. June 2017. MIT License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

class SubspaceNN(torch.nn.Module):
    def __init__(self, batch_size, input_dim, h_dim, output_dim, omega_dim=None):
        super(SubspaceNN, self).__init__()
        # param_meta maps each param to (dim1, dim2, initial_stdev)
        self.batch_size = batch_size
        self.omega_dim = omega_dim
        self.param_meta = {'W1': (input_dim, h_dim, 0.001), 'W2': (h_dim, h_dim, 0.001),
                      'W3': (h_dim, output_dim, 0.01), \
                      'b1': (1, h_dim, 0.0), 'b2': (1, h_dim, 0.0), 'b3': (1, output_dim, 0.0) }
                      
        self.names = [k for k in self.param_meta.keys()]
        self.counts = [self.param_meta[n][0]*self.param_meta[n][1] for n in self.names]
        self.slices = np.cumsum([0] + self.counts)
        self.theta_dim = int(self.slices[-1])
        
        self.init_param_space()
        print('model summary:\n\tthis model\'s omega space has {} parameters'.format(self.omega_dim))
        print('\tthis model\'s theta space has {} parameters'.format(self.theta_dim))
        
    def init_param_space(self):
        # initialize and concat
        flat_params = [np.random.randn(self.counts[i],1)*self.param_meta[n][2] for i, n in enumerate(self.names)]
        theta_init = np.concatenate(flat_params, axis=0)
        if self.omega_dim is None:
            self.flat_theta = nn.Parameter(torch.Tensor(theta_init), requires_grad=True)
        else:
            random_init = np.random.randn(self.theta_dim, self.omega_dim-1)

            # this is where the subspace magic happens
            A = np.concatenate((theta_init, random_init), axis=1) # first column is initializations
            p, _ = np.linalg.qr(A)
            self.P = Variable(torch.Tensor(p), requires_grad=False)

            omega = torch.zeros(self.omega_dim,1)
            omega[0] = theta_init[0,0] / p[0,0] # multiply the first column by this to get theta_init values
            self.omega = nn.Parameter(omega, requires_grad=True)
        
    def get_flat_theta(self):
        if self.omega_dim is None:
            return self.flat_theta
        else:
            return self.P.mm(self.omega).resize(self.theta_dim) # project from omega space to theta space

    def forward(self, X):
        flat_theta = self.get_flat_theta()
        thetas = {n: flat_theta[self.slices[i]:self.slices[i+1]] for i, n in enumerate(self.names)}
        thetas = {k : v.resize(self.param_meta[k][0], self.param_meta[k][1]) for k, v in thetas.items()}
        
        bs = X.size(0)
        h1 = F.relu(X.mm(thetas['W1']) + thetas['b1'].repeat(bs, 1))
        h2 = F.relu(h1.mm(thetas['W2']) + thetas['b2'].repeat(bs, 1))
        h3 = F.log_softmax(h2.mm(thetas['W3']) + thetas['b3'].repeat(bs, 1))
        return h3