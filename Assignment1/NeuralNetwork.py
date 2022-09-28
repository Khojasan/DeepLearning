# Package imports
import numpy as np
import sklearn
import sklearn.linear_model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Idk why I'm getting an error here
# %matplotlib inline

np.random.seed(1)

X, y = load_breast_cancer(return_X_y=True)
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size =0.2, random_state=0)

scaler = preprocessing.StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

trainx = train_data.T
trainy = train_labels.reshape(-1,1).T

testx = test_data.T
testy =test_labels.reshape(-1,1).T

trainx.shape, trainy.shape, testx.shape, testy.shape

X=trainx
Y=trainy

### START CODE HERE ###
shape_X = None
shape_Y = None
m = None  # training set size
### END CODE HERE ###

print ('No. of training samples: ' + str(m))
print ('Number of features per sample: ' + str(shape_X[0]))

def model_architecture(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    ### START CODE HERE ### 
    n_x = None # size of input layer
    n_h = None
    n_y = None # size of output layer
    ### END CODE HERE ###
    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    np.random.seed(2)

    
    ### START CODE HERE ###
    W1 = None
    b1 = None
    W2 = None
    b2 = None
    ### END CODE HERE ###
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def sigmoid(x):
    ### Update THE CODE HERE ###
    return None
    ### END CODE HERE ###

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### 
    W1 = None
    b1 = None
    W2 = None
    b2 = None
    ### END CODE HERE ###
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    ### START CODE HERE ### 
    Z1 = None
    A1 = None
    Z2 = None
    A2 = None
    ### END CODE HERE ###
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def compute_loss(A2, Y):
    """
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
       
    Returns:
    loss -- cross-entropy loss given equation (7)
    
    """
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy loss
    ### START CODE HERE ###
    logprobs = None
    loss = None
    ### END CODE HERE ###
    
    loss = float(np.squeeze(loss))

    assert(isinstance(loss, float))
    
    return loss

def backprop(parameters, cache, X, Y):
    """
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data
    Y -- "true" labels
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    ### START CODE HERE ### 
    W1 = None
    W2 = None
    ### END CODE HERE ###
        
    # Retrieve also A1 and A2 from dictionary "cache".
    ### START CODE HERE ### 
    A1 = None
    A2 = None
    ### END CODE HERE ###
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    ### START CODE HERE ### 
    dZ2 = None
    dW2 = None
    db2 = None
    dZ1 = None
    dW1 = None
    db1 = None
    ### END CODE HERE ###
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update(parameters, grads, learning_rate = 0.01):
    """
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    learning_rate -- The learning rate
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    ### START CODE HERE ### 
    W1 = None
    b1 = None
    W2 = None
    b2 = None
    ### END CODE HERE ###
    
    # Retrieve each gradient from the dictionary "grads"
    ### START CODE HERE ### 
    dW1 = None
    db1 = None
    dW2 = None
    db2 = None
    ## END CODE HERE ###
    
    # Update rule for each parameter
    ### START CODE HERE ### 
    W1 = None
    b1 = None
    W2 = None
    b2 = None
    ### END CODE HERE ###
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def NeuralNetwork(X, Y, n_h, num_iterations = 10000, learning_rate = 0.01, print_loss=False):
    """
    Arguments:
    X -- dataset
    Y -- labels 
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    learning_rate -- The learning rate
    print_loss -- if True, print the loss every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to make predictions.
    """
    
    np.random.seed(3)
    n_x = model_architecture(X, Y)[0]
    n_y = model_architecture(X, Y)[2]
    
    # Initialize parameters
    ### START CODE HERE ### 
    parameters = None
    ### END CODE HERE ###
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):
         
        ### START CODE HERE ### 
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = None
        
        # loss function. Inputs: "A2, Y, parameters". Outputs: "loss".
        loss = None
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = None
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters =  None
        
        ### END CODE HERE ###
        
        # Print the loss every 100 iterations
        if print_loss and i % 100 == 0:
            print ("loss after iteration %i: %f" %(i, loss))

    return parameters

def predict(parameters, X):
    """
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data 
    
    Returns
    predictions -- vector of predictions of our model
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    ### START CODE HERE ### 
    A2, cache = None
    predictions = None
    ### END CODE HERE ###
    
    return predictions

# Build a model with a n_h-dimensional hidden layer
parameters = None

# Print accuracy
predictions = None
print(accuracy_score(Y.T, predictions.T))