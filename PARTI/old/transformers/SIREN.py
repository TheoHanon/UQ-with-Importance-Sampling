import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm




class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, n_hidden_layers=3):
        super(MLP, self).__init__()
        
        layers = [nn.Linear(in_features, hidden_features), nn.ReLU()]
        
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_features, out_features))
        
        self.model = nn.Sequential(*layers)
        
        self.init_weights()
        
    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    layer.bias.data.fill_(0.0)
    
    def forward(self, x):
        x.requires_grad_(True)
        return self.model(x)
    

class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, omega_0 = 30, is_first = False, bias = True):

        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first   
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features, bias = bias)
        self.init_weights()


    def init_weights(self):
        with torch.no_grad():
            if (self.is_first):
                size = 1
                if size == 2:
                    self.linear.weight[...,0].uniform_( - 1/ self.in_features, 1 / self.in_features)
                    self.linear.weight[...,1].uniform_( - 4/ self.in_features, 4 / self.in_features)
                else:
                    self.linear.weight.uniform_( - 1/ self.in_features, 1 / self.in_features)

            else:
                self.linear.weight.uniform_(-np.sqrt(6 /  self.in_features) / self.omega_0, np.sqrt(6 / self.in_features)/ self.omega_0)


    def forward(self, input):
        return torch.sin(self.omega_0*self.linear(input))
   



class SIREN(nn.Module):
    
        def __init__(self, in_features, hidden_features,  out_features, first_omega_0, hidden_omega, nHiddenLayer = 1, last_linear = True):
            super().__init__()
            self.first_layer = SineLayer(in_features, hidden_features, first_omega_0, is_first=True)
    
            self.net = []
            
            for _ in range(nHiddenLayer):
                self.net.append(SineLayer(hidden_features, hidden_features, hidden_omega))
    
        
            if (last_linear):
                self.final_layer = nn.Linear(hidden_features, out_features, bias = False)
    
                with torch.no_grad():
                    self.final_layer.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega, 
                                                    np.sqrt(6 / hidden_features) / hidden_omega)
            else:
                self.final_layer = SineLayer(hidden_features, out_features, hidden_omega)
    
            self.net = nn.Sequential(*self.net)
    
        def forward(self, coords):

            coords.requires_grad_(True)
            x = self.first_layer(coords)
            x = self.net(x)
            return self.final_layer(x)
        


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (inputs,outputs,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing = 'xy'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def grad(outputs, inputs):
    # print(outputs, inputs)
    return torch.autograd.grad(outputs, [inputs], grad_outputs = torch.ones_like(outputs), create_graph = True)[0]
    
def hessian(y, x):
    ''' hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)
    '''
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = torch.autograd.grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = torch.autograd.grad(dydx[..., j], x, grad_y, create_graph=True)[0][..., :]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status


def laplace(outputs, inputs):
    gradient = grad(outputs, inputs)
    return divergence(gradient[...,0], inputs[...,0])


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def jacobian(y, x):
    ''' jacobian of y wrt x '''
    meta_batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1]).to(y.device) # (meta_batch_size*num_points, 2, 2)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = torch.autograd.grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status


def score_matching_loss(inputs, outputs):

    grad = torch.autograd.grad(
        outputs = outputs, 
        inputs = inputs, 
        grad_outputs = torch.ones_like(inputs),
        create_graph = True, 
        retain_graph = True,
    )[0]

    
    divergence = grad[..., 0]
    loss = outputs**2 + 2 * divergence

    return loss.mean()


def score_matching_loss_2(inputs, outputs, xi, D, dt):

    L = outputs + (2*D*dt)**(-0.5) * xi

    return L.pow(2).mean()




def train_siren(model, inputs, D, xi, dt, n_epoch=500, lr=1e-5):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    with tqdm(total = n_epoch) as pbar:
        for epochs in range(n_epoch):
            scores = model(inputs)
            # loss = score_matching_loss(inputs, scores)
            loss = score_matching_loss_2(inputs, scores, xi, D, dt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)


    model.eval()

    return model

def div_siren(inputs, outputs):
    du = torch.autograd.grad(
        outputs = outputs, 
        inputs = inputs, 
        grad_outputs = torch.ones_like(outputs),
        create_graph = True, 
        retain_graph = True,
    )[0]
    
    div = du[..., 0].view(-1, 1)
    
    return div


