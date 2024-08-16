import torch

class RobustUniform(torch.distributions.Uniform):

    def __init__(self, low: float, high: float, validate_args=None, val_max = 1e2) :

        super(RobustUniform, self).__init__(low, high, validate_args = validate_args)
        self.val_max = val_max

    
    def log_prob(self, value):
        
        with torch.no_grad():
            in_bounds = (value >= self.low) & (value <= self.high)

        log_prob = 0.0#super(RobustUniform, self).log_prob(value)
        log_prob = torch.where(in_bounds, log_prob, -self.val_max)

        return log_prob
    
    def _validate_sample(self, value):
        pass
    
