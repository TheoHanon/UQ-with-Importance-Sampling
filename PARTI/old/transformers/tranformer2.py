import torch
import torch.nn as nn
import numpy as np  
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class ParticleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc = nn.Linear(d_model, input_size)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer(src, src)
        output = self.fc(output)
        return output
    
def score_matching_loss_2(inputs, outputs, xi, D, dt):
    L = outputs + (2*D*dt)**(-0.5) * xi

    return L.pow(2).mean()

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

# Parameters
N = 1000  # Number of particles
n = 1  # Size of each particle vector
d_model = 64
nhead = 4
num_encoder_layers = 3
dim_feedforward = 256
dropout = 0.1
batch_size = 512
num_epochs = 100
learning_rate = 0.001


xi = torch.randn(N, n)
D = 1
dt = 1

x = np.sqrt(2*D*dt) * xi
x.requires_grad = True


h = TransformerModel(input_size=n, d_model=d_model, nhead=nhead, 
                         num_encoder_layers=num_encoder_layers, 
                         dim_feedforward=dim_feedforward, dropout=dropout)



dataset = torch.utils.data.TensorDataset(x, xi)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer

optimizer = optim.Adam(h.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    h.train()
    total_loss = 0
    for batch in dataloader:
        x_batch, xi_batch = batch
        optimizer.zero_grad()
        outputs = h(x_batch.unsqueeze(1))  # Add sequence dimension
        loss = score_matching_loss_2(x_batch, outputs, xi_batch, D, dt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}')

print("Training complete")

h.eval()
x = torch.linspace(-1, 1, 1000).view(-1, 1, n).requires_grad_(True)
y = h(x)

div_h = div_transformer(x, y.unsqueeze(1))



x = x.squeeze().detach().numpy()
y = y.squeeze().detach().numpy()
div_h = div_h.squeeze().detach().numpy()

import matplotlib.pyplot as plt

plt.plot(x, y)
plt.show()









