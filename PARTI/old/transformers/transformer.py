import torch 
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

def reset_weights(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_head, hidden_dim):
        super(EncoderBlock, self).__init__()

        self.attention = nn.MultiheadAttention(d_model, num_head, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Softplus(beta=7.5),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x):
        
        atn, _ = self.attention(x, x, x)
        x = x + atn
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)

        return x
    

class MLPEmbedding(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPEmbedding, self).__init__()

        self.input_dim = input_dim # = n
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim # = model_d

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Softplus(beta = 7.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, N, n)
        """

        return self.mlp(x) # (batch_size, N, model_d)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer):
        super(Decoder, self).__init__()
        layers = []

        for _ in range(n_layer):
            layers.append(nn.Linear(hidden_dim if len(layers) else input_dim, hidden_dim))
            layers.append(nn.Softplus(beta = 7.5))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
            

    def forward(self, x):
        return self.mlp(x)
    
 
class Transformer(nn.Module):

    def __init__(self, input_dim, hidden_dim_embedding, d_model, num_head, n_encoder, hidden_dim_encoder, n_layer_decoder, hidden_dim_decoder, output_dim=1):
        super(Transformer, self).__init__()

        self.embedding = MLPEmbedding(input_dim, hidden_dim_embedding, d_model)
        # self.embedding = nn.Linear(input_dim, d_model)
        self.encoder = nn.Sequential(*[EncoderBlock(d_model, num_head, hidden_dim_encoder) for _ in range(n_encoder)])
        self.decoder = Decoder(d_model, hidden_dim_decoder, output_dim, n_layer_decoder)
        # self.decoder = nn.Linear(d_model, output_dim)


    def forward(self, x):

        x = self.embedding(x)
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    



class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layer):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        layers = []

        for _ in range(n_layer):
            layers.append(nn.Linear(hidden_dim if len(layers) else input_dim, hidden_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        

    def forward(self, x):
        return self.mlp(x)
    

class Line(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Line, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mlp = nn.Linear(input_dim, output_dim, bias = True)


    def forward(self, x):
        return -self.mlp(x)



def fit_loss(inputs, outputs):

    L = (inputs**2 - outputs)

    return L.pow(2).mean()
        

def score_matching_loss(inputs, outputs, beta = 1e-3):

    grad = torch.autograd.grad(
        outputs = outputs, 
        inputs = inputs, 
        grad_outputs = torch.ones_like(outputs),
        create_graph = True, 
        retain_graph = True,
    )[0]

    squared_norm = 0.5 * (outputs**2)#.sum(dim = 1)
    trace_jacobian = grad#.sum(dim = 1)

    
    loss = squared_norm + trace_jacobian 
    
    return loss.mean() 


def minimum_gradient_loss(inputs, outputs):

    
    grad = torch.autograd.grad(
        outputs = outputs, 
        inputs = inputs, 
        grad_outputs = torch.ones_like(outputs),
        create_graph = True, 
        retain_graph = True,
    )[0]
    
    return grad.pow(2).mean()

def score_matching_loss_2(inputs, outputs, xi, D, dt):
    
    L = outputs + (2*D*dt)**(-0.5) * xi
    return L.pow(2).mean() 



def train_v2(model, data, n_epochs, xi, D, dt, max_steps = 50000, clip_grad = 1.0):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = n_epochs, eta_min = 0)

    with tqdm(total = n_epochs) as pbar:
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = model(data)
            # loss = score_matching_loss_2(data, outputs, xi, D, dt)
            loss = score_matching_loss(data, outputs)
            # loss = fit_loss(data, outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = clip_grad)
            optimizer.step()
            # scheduler.step()
            pbar.set_postfix({'loss': loss.item()})
            pbar.update(1)
    
    model.eval()
    return model


def div_transformer(inputs, outputs):

    du = torch.autograd.grad(
        outputs = outputs, 
        inputs = inputs, 
        grad_outputs = torch.ones_like(outputs),
        create_graph = False, 
        retain_graph = False,
    )[0]
    
    div = du[..., 0].view(-1, 1)
    
    return div


# if __name__ == "__main__":

#     # Setup a dummy examples 

#     batch_size = 1
#     N = 1000
#     n = 1
#     x = torch.randn(batch_size, N, n)
#     # x = torch.linspace(-1, 1, N).view(1, -1, n)

#     # The goal is to learn the score function grad_x(log q) where q ~ N(0, 1)

#     # model = Transformer(input_dim=n, hidden_dim=32, d_model=32, num_head=2, d_phi=32, L=4)
#     # model = Transformer(input_dim=n, hidden_dim_embedding=32, d_model=64, num_head=2, n_encoder=2, hidden_dim_encoder=256, n_layer_decoder=1, hidden_dim_decoder=64, output_dim=1)
#     model = MLP(input_dim=1, hidden_dim=256, output_dim=1, n_layer=3)
#     data = x.requires_grad_(True)    

#     model = train_v2(model, data, n_epochs=10000, xi = torch.randn(N, 1), D=1, dt=1, max_steps=20, clip_grad=1.0)
    
#     xplot1 = torch.linspace(-1, 1, 1000).view(1, -1, 1).requires_grad_(True)
#     div1 = div_transformer(xplot1, model(xplot1)).squeeze().detach().numpy()
#     yplot1 = model(xplot1).squeeze().detach().numpy()
#     xplot1 = xplot1.squeeze().detach().numpy()
   

#     plt.plot(xplot1, yplot1)
#     plt.plot(xplot1, -xplot1, color='red', linestyle='--')
#     plt.plot(xplot1, div1, color='green', linestyle='--')
#     plt.show()
    







    