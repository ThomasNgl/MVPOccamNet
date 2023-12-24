import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class OccamNet(nn.Module):
    def __init__(self, input_dims, output_dims, activation_layers, temperatures, skip_connections = False, is_none = False):
        super(OccamNet, self).__init__()
        self.input_dim = len(input_dims)
        self.output_dim = len(output_dims)

        self.activation_layers = activation_layers
        self.temperatures = temperatures

        self.skip_connections = skip_connections
        self.is_none = int(is_none)

        # For loop to initialize the network
        self.w_layers = []
        current_dim = self.input_dim + self.is_none
        for a_layer in self.activation_layers[:-1]:
            arity_layer = 0
            for base_function in a_layer:
                arity_layer += base_function.arity 
            self.initialize_w_layer(current_dim, arity_layer)
            nb_bases = len(a_layer) 
            if self.skip_connections:
                current_dim += nb_bases
            else: 
                current_dim = nb_bases
        # -self.is_none because we don't need to add a None as input to the last layer
        # in case of skip connection
        if self.skip_connections:
            current_dim -= self.is_none
        self.initialize_w_layer(current_dim, self.output_dim)

        self.w_layers = nn.ParameterList(self.w_layers)
    
        self.path_to_a_index, self.index_to_path_index = self.construct_index_path_matrices()

        self.sample_size = torch.Size([1])

    # Private 
    ###
    def initialize_w_layer(self, input_dim, output_dim):
        # Initialize a layer in the network
        w = nn.Parameter(torch.ones(output_dim, input_dim, dtype=torch.float64))
        self.w_layers.append(w)

    def construct_index_path_matrices(self):
        # construct path_to_index et index_to_path matrices
        path_to_a_index = []
        index_to_path_index = []
        # First the input vector
        j = 0
        index_layer = []
        for i in range(0,self.input_dim + self.is_none):
            index_layer.append(j)
            j += 1
        path_to_a_index.append(index_layer)
        # Second the activation layers
        for a_layer in self.activation_layers:
            index_layer = []
            index_to_path_layer = []
            k = 0
            for base_f in a_layer:
                arity = base_f.arity
                path_indices = list(range(k, k+arity))
                index_to_path_layer.append(path_indices)
                k += arity
                index_layer.append(j)
                j += 1
            path_to_a_index.append(index_layer)
            index_to_path_index.append(index_to_path_layer)
        return path_to_a_index, index_to_path_index

    def find_value_index(self, matrix, value):
        # tool function to find the index of a value in a matrix 
        for i, row in enumerate(matrix):
            for j, element in enumerate(row):
                if element == value:
                    return i, j
        return None

    def apply_softmax_layer(self, w_layer, T):
        # Apply softmax on w layer to have a p layer which is a torch.distributions
        p_layer = F.softmax(w_layer/T, dim = 1)
        return p_layer
   
    def get_the_path(self, the_sub_path, top_key, the_keys_path, the_keys_path2):

        layer_index, input_index = self.find_value_index(self.path_to_a_index, top_key)
        if layer_index == 0:
            the_sub_path.append(input_index)
            return 
        else:
            # The base function
            base_f = self.activation_layers[layer_index-1][input_index]
            the_sub_path.append(base_f)
            # The position of the inputs of the base f in its layer
            # len of rows is arity of base_f
            rows = self.index_to_path_index[layer_index-1][input_index]

            next_T = self.temperatures[layer_index-1]
            # Select the rows that correspond to the inputs od the base f
            # shape (arity, nb possible inputs)
            next_w_layer = self.w_layers[layer_index-1][rows]
            the_keys_path.append([layer_index, rows])

            next_p_layer = self.apply_softmax_layer(next_w_layer, next_T)
            next_p_layer = torch.distributions.Categorical(next_p_layer)
            # shape (arity, 1)
            next_top_keys = next_p_layer.sample(self.sample_size)
            the_keys_path2.append([k.item() for k in copy.deepcopy(next_top_keys[0])])
            # shape (arity, nb possible inputs)
            #next_top_ps, next_top_keys = torch.topk(last_p_layer, k=1, dim=1)
            #  Shape (arity)
            for top_key in next_top_keys[0]:
                self.get_the_path(the_sub_path, top_key, the_keys_path, the_keys_path2)
    ###
    
    # Public (may be used in forward)
    ###
    def samples_paths_and_probas(self, nb_sampled_paths):
        # Find n paths by selecting for each base function the input index (indices if arity>1) 
        the_paths = []
        the_keys_paths = []
        the_keys_paths2 = []
        last_T = self.temperatures[-1] 
        last_a_layer = self.activation_layers[-1]
        nb_layers = len(self.activation_layers)
        for _ in range(0, nb_sampled_paths):
            proba = 1
            # shape (output dim)
            the_path = [[final_a] for final_a in last_a_layer]
            the_keys_path = [[nb_layers, [i]] for i in range(len(last_a_layer))]
            # shape (sum arities == output dim, nb possible inputs)
            last_w_layer = self.w_layers[-1]
            last_p_layer = self.apply_softmax_layer(last_w_layer, last_T)
            last_p_layer = torch.distributions.Categorical(last_p_layer)
            # shape (sum of arities == output dim, 1)
            initial_top_keys = last_p_layer.sample(self.sample_size)
            the_keys_path2 = [[key.item()] for key in copy.deepcopy(initial_top_keys[0])]
            #top_ps, top_keys = torch.topk(last_p_layer, k=1, dim=1)
            for the_sub_path, top_key in zip(the_path, initial_top_keys[0]):
                self.get_the_path(the_sub_path, top_key, the_keys_path, the_keys_path2)
            the_paths.append(the_path)
            the_keys_paths.append(the_keys_path)
            the_keys_paths2.append(copy.deepcopy(the_keys_path2))

        return the_paths, the_keys_paths, the_keys_paths2
    
    def get_proba(self, the_keys_paths, the_keys_paths2):
        # Get the probabilities of specific chosen paths
        nb_samples = len(the_keys_paths)
        probas = []
        for sample_index in range(0, nb_samples):
            proba = 1
            for position, k in zip(the_keys_paths[sample_index], the_keys_paths2[sample_index]):
                layer_index, rows = position[0], position[1]
                w_layer = self.w_layers[layer_index-1][rows]
                p_layer = self.apply_softmax_layer(w_layer, self.temperatures[layer_index-1])
                p_layer = torch.distributions.Categorical(p_layer)
                prob_layer = torch.exp(p_layer.log_prob(torch.tensor([k])))
                proba *= torch.prod(prob_layer[0], dim = -1)
            probas.append(proba)
        probas = torch.stack(probas, dim=0)
        return probas
    
    def get_proba_distributions(self):
        # Get the porba distributions for each base function for each layer of the network
        proba_distributions = []
        for w_layer, T in zip(self.w_layers, self.temperatures):
            p_layer = self.apply_softmax_layer(w_layer, T)
            p_layer = torch.distributions.Categorical(p_layer)

            # .probs is an attribute of torch.distribution
            proba_distributions.append(p_layer.probs)
        return proba_distributions

    def forward(self, x, the_paths):
        nb_sampled_paths = len(the_paths)
        # Apply the full network on an element x following the paths 
        # Thus return y of shape == (nb sampled paths, output dim)
        y = []
        for sample_index in range(0, nb_sampled_paths):
            the_path = the_paths[sample_index]
            # One sub path for each output dim
            sample_y = []
            for sub_path in the_path:
                stored_inputs = []
                for element in reversed(sub_path):
                    if type(element) is int:
                        stored_inputs.append(x[sample_index][element])
                    elif element.key == 'ide':               
                        output = stored_inputs[-1]
                    else:
                        input_to_base_f = reversed(stored_inputs[-element.arity: ])
                        stored_inputs = stored_inputs[: -element.arity]
                        stored_inputs.append(element.forward(*input_to_base_f))
                sample_y.append(output)
            y.append(sample_y)
        return y
    ###    

