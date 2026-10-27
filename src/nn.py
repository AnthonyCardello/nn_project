import numpy as np

def Generate_nn(inp_size: int, hl: np.ndarray) -> list[np.ndarray]:
    # make layer matrixes by  inputsize high(like num of rows) and number of hidden nodes dimeations
    rng = np.random.default_rng(42) # you use a normal distrobution cause you want most of the random starting values close to zero this kinda makes less spikes in ur weights (like a uniform dist would) and thus this smoother random starting point has been show to be better for gradient descent 
    nn_empty = [] # it would be faster to make a big np array instead of storing each layer in a python list but then at that point maybe I should just write it in c++
    for dim in hl:
        nn_empty.append(rng.standard_normal((inp_size,dim)))
    return nn_empty;



