import numpy as np


def relu(z):
    return np.maximum(0,z)

def sigmoid(z):
    return 1.0/ (1.0 +np.exp(-z))

def Generate_nn(shape: np.ndarray) -> list[list[np.ndarray]]:# i thought type annotations where a neat idea but idk if its really worth it readablity wise since it does almost nothing without a type checker
    # make layer matrixes by  inputsize high(like num of rows) and number of hidden nodes dimeations
    rng = np.random.default_rng(42) # you use a normal distrobution cause you want most of the random starting values close to zero this kinda makes less spikes in ur weights (like a uniform dist would) and thus this smoother random starting point has been show to be better for gradient descent 
    nn_empty:list[list[np.ndarray]] = [] # it would be faster to make a big np array instead of storing each layer in a python list but then at that point maybe I should just write it in c++
    for l_inputs,l_nodes in zip(shape, shape[1:]):
        layer = rng.standard_normal((l_inputs,l_nodes)) * np.sqrt(2/l_inputs) # you divide by sqrt(input_size) to control the varience, cause in a neuron ur gonna be adding input_size amount of weights, with each addition that increases the varience by 1 in the function which is why u divide (its sqrt cause varience is squared), and the 2 comes from some paper saying I should.
        nn_empty.append([layer,np.zeros(l_nodes)])
    return nn_empty

def forward(nn_start: list[list[np.ndarray]], X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    A: list[np.ndarray] = []
    Z: list[np.ndarray] = []
    i = 0
    for layer in nn_start:
        if(i < 1): Z.append(X.dot(layer[0])+ layer[1]) # layer[1] is the bias
        else: Z.append(Z[i-1].dot(layer[0])+ layer[1])
        if i != (len(nn_start) - 1):
            A.append(relu(Z[i]))
        else:
            A.append(sigmoid(Z[i]))
        i += 1
    return A,Z

# def cost()  gonna use some binary cross entrophy kinky function

def loss(A: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return -1/(Y.shape[0])* np.sum(Y * (np.log(A))) + ((1-Y)*np.log((1-A)))





