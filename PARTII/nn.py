from torch import nn
import torch


class Linear(nn.Module):

    def __init__(self, input_features : int, output_features : int):
            
            super(Linear, self).__init__()
    
            self.input_features = input_features
            self.output_features = output_features
    
    def forward(self, x : torch.Tensor, W : torch.Tensor) -> torch.Tensor:
        """
        Args:
        x : torch.Tensor, size (B, M, input_features) : Input
        W : torch.Tensor, size (B, input_features*output_features + output_features) : Weights and bias

        output:
        torch.Tensor, size (B, M, output_features) : Output
        """
        
        weights = W[..., :self.input_features*self.output_features].view(-1, self.output_features, self.input_features)
        bias = W[..., self.input_features*self.output_features:].view(-1, self.output_features)

        return torch.bmm(x, weights.transpose(1, 2)) + bias.unsqueeze(1)


    
class NN(nn.Module):

    def __init__(self, n: int, m : int, hidden_size : int, n_hidden : int = 1):

        super(NN, self).__init__()

        self.nn = nn.ModuleList()

        self.nn.append(Linear(n, hidden_size))

        for i in range(n_hidden):
            self.nn.append(Linear(hidden_size, hidden_size))

        self.nn.append(Linear(hidden_size, m))


    def get_params_count(self) -> int:
        return sum([layer.input_features*layer.output_features + layer.output_features for layer in self.nn])
    

    def forward(self, x : torch.Tensor, W : torch.Tensor) -> torch.Tensor:
        
        idx = 0
        x = x.expand(W.shape[0], -1, -1)

        for layer in self.nn[:-1]:
            w = W[..., idx:idx + layer.input_features*layer.output_features + layer.output_features]
            # x = nn.functional.gelu(layer(x, W = w))        
            x = torch.sin(layer(x, W = w))
            idx += layer.input_features*layer.output_features + layer.output_features

        w  = W[..., idx: idx + self.nn[-1].input_features*self.nn[-1].output_features + self.nn[-1].output_features]

        return self.nn[-1](x, W=w)
    