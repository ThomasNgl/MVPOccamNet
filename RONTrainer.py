import numpy as np
import torch
import torch.nn as nn
import copy

class RONTrainer(nn.Module):
    def __init__(self, recurrent_model, criterion, optimizer):
        super(RONTrainer, self).__init__()
        self.r_model = recurrent_model
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
        if self.r_model.is_none:
            X = [x.append(np.array([None])) for x in X]
        X = [[copy.deepcopy(x) for _ in range(nb_sampled_paths)] for x in X]

        # Reshape Y so that shape == (nb elements, output dim).
        Y = torch.tensor(Y, requires_grad = False)
        #hidden_dim = self.r_model.hidden_dim
        for epoch in range(max_epochs):
            print('epoch', epoch)
            self.optimizer.zero_grad()
            # First step is to sample paths from the network and their corresponding proba
            the_paths, the_keys_probas, the_keys_path2 = self.r_model.samples_paths_and_probas(nb_sampled_paths)
            the_probas = self.r_model.get_proba(the_keys_probas, the_keys_path2)
            loss = 0
            # Do the rolling out/forward 
            pred_list, h_list = self.r_model.forward(X, the_paths) 

            # (nb elements, nb sampled paths, output dim), (nb elements, nb sampled paths, hidden dim), (nb sampled paths, 1)
            # Intervert nb sampled paths and nb elements

            for pred, target in zip(pred_list, Y):
                loss =+ self.criterion.forward(pred, target, the_probas)
            #???? If I don't use retain_graph = True I have an error
            loss.backward(retain_graph=True)
            for param in self.r_model.parameters():
                print(param, param.grad)
            self.optimizer.step()
        # Return the trained model
        return self.r_model
