import numpy as np
import torch
import torch.nn as nn
import copy

class ONTrainer(nn.Module):
    def __init__(self, model, criterion, optimizer):
        super(ONTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def fit(self, X, Y, max_epochs, nb_sampled_paths):
        '''
        X :np.array like (nb elements, input dim)
        Don't need to be a tensor as no grad through X
        Y :torch.tensor (nb elements, output dim)
        max_epochs: int, the max number of epochs for the training
        nb_sampled_paths : int, the number of sampled paths

        '''
        # Reshape X so that shape == (nb elements, nb sampled paths, input dim).
        # input dim + 1 if a None is added.
        if self.model.is_none:
            X = [x.append(np.array([None])) for x in X]
        X = [[copy.deepcopy(x) for _ in range(nb_sampled_paths)] for x in X]

        # Reshape Y so that shape == (nb elements, output dim).
        Y = torch.tensor(Y, requires_grad = False)

        for epoch in range(max_epochs):
            print('epoch', epoch)
            self.optimizer.zero_grad()
            # First step is to sample paths from the network and their corresponding proba
            the_paths, the_keys_probas, the_keys_path2 = self.model.samples_paths_and_probas(nb_sampled_paths)
            the_probas = self.model.get_proba(the_keys_probas, the_keys_path2)
            loss = 0
            # Second step is to compute the prediction of the model for each element of X
            for x, y in zip(X, Y):
                # This imply to compute the pred of the model for each sampled path
                pred = self.model.forward(x, the_paths) # shape == (nb sampled paths, output dim)
                loss += self.criterion.forward(pred, y, the_probas)
            #???? If I don't use retain_graph = True I have an error
            loss.backward(retain_graph=True)
            # for param in self.model.parameters():
            #     print(param, param.grad)
            self.optimizer.step()
        # Return the trained model
        return self.model
