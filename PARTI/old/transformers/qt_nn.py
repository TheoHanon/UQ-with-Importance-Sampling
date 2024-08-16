import PARTI.transformers.SIREN as sr
import PARTI.transformers.transformer as tr
import torch
from torch import nn
from tqdm.autonotebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



if __name__ == "__main__":
    xi = torch.randn(1000, 1)
    D = 1
    dt = 1


    f = lambda x: -1 * np.ones_like(x)
    x = np.sqrt(2*D*dt) * xi
    x.requires_grad = True


    h = tr.Transformer(1, 16, 256, 2, 2, 2)
    # h = sr.SIREN(1, 256, 1, 1, 30, 3)

    # h = sr.train_siren(model = h, inputs=x, n_epoch= 1000, lr = 1e-4, dt = dt, D = D, xi = xi)
    h = tr.train(h, x, D, xi, dt, n_epoch=1000)
    # h.eval()

    x = torch.linspace(-1, 1, 1000).unsqueeze(-1).requires_grad_(True)
    x = x.unsqueeze(0)
    y = h(x)

    div_h = tr.div_transformer(x, y)


    x = x.squeeze().detach().numpy()
    y = y.squeeze().detach().numpy()
    div_h = div_h.squeeze().detach().numpy()

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    div_h = div_h[idx]
 
    fig, ax  = plt.subplots()
    ax.plot(x, y)
    
    ax.twinx().plot(x, div_h, color='red')

    plt.show()


