import numpy as np
import torch
import torch.nn as nn
from OccamNet import OccamNet
import copy

class RecurrentOccamNet(OccamNet):
    def __init__(self, input_dims, hidden_dim, output_dims, 
                 activation_layers, temperatures,
                 skip_connections = False, is_none = False):
        
        self.hidden_dim = hidden_dim

        input_dims.append(hidden_dim)
        output_dims.append(hidden_dim)
        super(RecurrentOccamNet, self).__init__(input_dims, output_dims, 
                             activation_layers, temperatures, 
                            skip_connections, is_none)
        
    
    def forward(self, X, the_paths, h0 = None):
        # X shape (nb elements, nb sampled paths, input dim)
        nb_sampled_paths = len(the_paths)
        if h0 is None:
            h = 0.5 * np.ones((nb_sampled_paths, self.hidden_dim))

        h_sequence = []
        y_sequence = []
        i=0
        for x in X:
            i+=1
            # x shape (nb sampled paths, input dim)
            # h shape (hidden dim)
            current_x = copy.deepcopy(x)
            for sample_index in range(0, nb_sampled_paths):
                current_x[sample_index].append(copy.deepcopy(h[sample_index]))
            # current_x shape (nb sampled paths, input dim + hidden dim)

            pred = super().forward(current_x, the_paths)
            # pred shape (nb sampled paths, output dim + hidden dim)
            y = []
            for sample_index, sample_pred in enumerate(pred):
                y.append(sample_pred[0])
                h_dim = sample_pred[-1].shape[0]
                if h_dim == self.hidden_dim or h_dim == 1:
                    h[sample_index] += sample_pred[-1]

            y_sequence.append(copy.deepcopy(y))
            h_sequence.append(copy.deepcopy(h))
        return y_sequence, h_sequence