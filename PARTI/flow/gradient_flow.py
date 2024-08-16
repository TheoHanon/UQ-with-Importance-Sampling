import torch
from tqdm.autonotebook import tqdm
from flow.flow import Flow


class GradientFlow(Flow):

    def flow(self) -> None:

        x = torch.empty((self.nStep, self.N, self.model.n), dtype=torch.float32)
        x[0] = self.q0.sample((self.N, self.model.n))

        with tqdm(total=self.nStep, desc="Gradient flow") as pbar:
            for i in range(self.nStep - 1):
                gradV = - self.model.grad_logP_XY(x[i], numpy = False)
                x[i + 1] = x[i] - self.dt * gradV
                pbar.update(1)

        self.x = x

        return 

    def logQ(self) -> None:

        if self.x is None:
            self.flow()

        xt = self.x

        logQ = torch.empty((self.nStep, self.N), dtype=torch.float32)
        logQ[0] = self.q0.log_prob(xt[0]).sum(dim = 1)

        laplace_logP = self.model.laplace_logP_XY(xt, numpy = False).squeeze()
        for i in range(1, self.nStep):

            logQ[i] = logQ[i-1] - self.dt * (laplace_logP[i-1] + laplace_logP[i]) / 2

        self.logq = logQ

        return