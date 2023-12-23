import torch
import torch.nn as nn
import math
import numpy as np

class Loss(nn.Module):
    def __init__(self, std):
        super(Loss, self).__init__()
        self.std = std

        # Compute C and B ones, not each time method fitness is called
        self.C = 1/(np.sqrt(2*math.pi)*self.std)
        self.B = (2*(self.std**2))

    def fitness(self, prediction, target):
        nb_sampled_paths = len(prediction)
        target_dim = len(target)
        fits = []
        for sample_index in range(nb_sampled_paths):
            if len(prediction[sample_index]) != target_dim or np.any(prediction[sample_index] == None):
                fit = torch.tensor(0., dtype=torch.float64)
            else:
                #print(prediction[sample_index], target)
                pred = np.array(prediction[sample_index])
                pred = torch.tensor(pred, requires_grad = False)
                mse = (pred - target).clone().detach()**2
                fit = self.C*np.exp(-mse/self.B).mean()

                #fit = np.mean(self.C*np.exp(-mse/self.B))
            fits.append(fit)
        return torch.tensor(np.array(fits), requires_grad = False)
        
    def forward(self, 
                prediction,
                target, 
                the_probas):
        
        fit = self.fitness(prediction, target)
        # shape = (nb samples, 1)
        #prob = net.get_proba(the_keys_path, the_keys_path2)
        log_prob = torch.log(the_probas)
        loss = -log_prob * fit
        mean_loss = loss.mean()

        # mean loss is a real value
        return mean_loss



