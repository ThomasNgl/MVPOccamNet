import numpy as np
import warnings

class on_base:
    def __init__(self):
        self.arity = 1
        self.key = 'base'

    def get_dim(self, x_dim):
        return x_dim
    
    def apply_base(self, function, x):
        if np.any(x==None):
            #return  np.array([None])
            return None

        else:
            return function(x)
        
class on_base_2(on_base):
    def __init__(self):
        super(on_base_2, self).__init__()
        self.arity += 1

    def get_dim(self, x0_dim, x1_dim):
        if  x0_dim[0] > x1_dim[0]: 
            return x0_dim
        else:
            return x1_dim

    def apply_base_2(self, function, x0, x1):
        if np.any(x0==None) or np.any(x1==None):
            return np.array([None])
            #return None
        else:
            # warnings.filterwarnings("error", category=RuntimeWarning)
            # try:
            #     result = function(x0, x1)
            # except RuntimeWarning as rw:
            #     print(f"Caught a RuntimeWarning: {rw}", x0, x1)
            # return result
            return function(x0, x1)
        
class on_cos(on_base):
    def __init__(self):
        super(on_cos, self).__init__()
        self.key = 'cos'

    def forward(self, x):
        return self.apply_base(np.cos, x)
    
class on_sin(on_base):
    def __init__(self):
        super(on_sin, self).__init__()
        self.key = 'sin'

    def forward(self, x):
        return self.apply_base(np.sin, x)

class on_identity(on_base):
    def __init__(self):
        super(on_identity, self).__init__()
        self.key = 'ide'
   
    def forward(self, x):
        return self.apply_base(identity, x)
    
def identity(x):
    return x

class on_sum(on_base_2):
    def __init__(self):
        super(on_sum, self).__init__()
        self.key = 'sum'

    def forward(self, x0, x1):
        return self.apply_base_2(np.add, x0, x1)

class on_sub(on_base_2):
    def __init__(self):
        super(on_sub, self).__init__()
        self.key = 'sub'

    def forward(self, x0, x1):
        return self.apply_base_2(np.subtract, x0, x1)

class on_mul(on_base_2):
    def __init__(self):
        super(on_mul, self).__init__()
        self.key = 'mul'
    
    def forward(self, x0, x1):
        return self.apply_base_2(np.multiply, x0, x1)

class on_div(on_base_2):
    def __init__(self):
        super(on_div, self).__init__()
        self.key = 'div'
    
    def forward(self, x0, x1):
        if np.any(x1==0):
            return np.array([None])
        return self.apply_base_2(np.divide, x0, x1)

class on_dot(on_base_2):
    def __init__(self):
        super(on_dot, self).__init__()
        self.key = 'dot'

    def get_dim(self, x0_dim, x1_dim):
        return [0, 1]

    def forward(self, x0, x1):
        return self.apply_base_2(dot_product, x0, x1)

def dot_product(x0, x1):
    return np.array([sum(x0 * x1)])
        


