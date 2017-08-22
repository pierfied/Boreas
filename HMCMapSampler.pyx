from abc import abstractmethod
from pyhmc import hmc

class HMCMapSampler:
    def __init__(self,y_map):
        self.y0 = y_map.copy()

    def sample(self,n_samples,epsilon):
        # Run the chain.
        self.samples = hmc(self.log_prob,self.y0,n_samples,epsilon=epsilon)

        return self.samples

    @abstractmethod
    def log_prob(self,y_map):
        pass