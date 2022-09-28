import numpy as np
import os
import time
import timeit
import torch
import math

# Section 2

# Question 1

def inefficient_dot(x, y):
    """
    Inefficient dot product of two arrays.

    Parameters:
    x (numpy.ndarray): 1-dimensional numpy array.
    y (numpy.ndarray): 1-dimensional numpy array.
    
    Returns:
    numpy.int64: scalar quantity.
    """

    assert(len(x) == len(y))

    result = 0
    for i in range(len(x)):
        result += x[i]*y[i]

    return result

def dot(x, y):
    """
    Dot product of two arrays.

    Parameters:
    x (numpy.ndarray): 1-dimensional numpy array.
    y (numpy.ndarray): 1-dimensional numpy array.

    Returns:
    numpy.int64: scalar quantity.
    """

    ndot = np.dot(x,y)

    return ndot

np.random.seed(0)
X = np.random.randint(-1000, 1000, size=3000)
Y = np.random.randint(-1000, 1000, size=3000)

tic = time.process_time() 
idot = inefficient_dot(X,Y)
toc = time.process_time() 

print("Inefficient dot product = "+ str(idot)); 
print("Computation time = " + str(1000*(toc - tic )) + "ms") 

n_tic = time.process_time() 
ndot = dot(X,Y) 
n_toc = time.process_time() 
print("\nEfficient dot product = "+str(ndot)) 
print("Computation time = "+str(1000*(n_toc - n_tic ))+"ms") 


# Question 2

def inefficient_outer(x, y):
    """
    Inefficiently compute the outer product of two vectors.

    Parameters: 
    x (numpy.ndarray): 1-dimensional numpy array.
    y (numpy.ndarray): 1-dimensional numpy array.

    Returns: 
    numpy.ndarray: 2-dimensional numpy array.
    """
    result = np.zeros((len(x), len(y))) 
    for i in range(len(x)):
        for j in range(len(y)):
            result[i, j] = x[i]*y[j]
    
    return result

def outer(x, y):
    """
    Compute the outer product of two vectors.

    Parameters: 
    x (numpy.ndarray): 1-dimensional numpy array.
    y (numpy.ndarray): 1-dimensional numpy array.

    Returns: 
    numpy.ndarray: 2-dimensional numpy array.
    """

    nouter = np.outer(x,y)
    return nouter

np.random.seed(0)
X = np.random.randint(-1000, 1000, size=3000)
Y = np.random.randint(-1000, 1000, size=3000)
tic = time.process_time() 
iouter = inefficient_outer(X,Y)
toc = time.process_time() 
print("Inefficient Outer Product = \n"+ str(iouter)); 
print("Computation time = "+str(1000*(toc - tic ))+"ms") 
n_tic = time.process_time() 
nouter = outer(X,Y)
n_toc = time.process_time() 
print("Efficient Outer Product = \n"+ str(nouter)); 
print("Computation time = "+str(1000*(n_toc - n_tic ))+"ms") 


# Question 3

def inefficient_multiply(x, y):
    """
    Inefficiently multiply arguments element-wise.

    Parameters: 
    x (numpy.ndarray): 1-dimensional numpy array.
    y (numpy.ndarray): 1-dimensional numpy array.

    Returns: 
    numpy.ndarray: 1-dimensional numpy array.
    """
    assert(len(x) == len(y))
    
    result = np.zeros(len(x))
    for i in range(len(x)):
        result[i] = x[i]*y[i]
    
    return result

def multiply(x, y):
    """
    Multiply arguments element-wise.

    Parameters: 
    x (numpy.ndarray): 1-dimensional numpy array.
    y (numpy.ndarray): 1-dimensional numpy array.

    Returns: 
    numpy.ndarray: 1-dimensional numpy array.
    """
    
    nmultiply = np.multiply(x,y)

    return nmultiply

np.random.seed(0)
X = np.random.randint(-1000, 1000, size=3000)
Y = np.random.randint(-1000, 1000, size=3000)

tic = time.process_time() 
imultiply = inefficient_multiply(X, Y) 
toc = time.process_time() 
print("Inefficient Element wise Product =\n "+ str(imultiply)); 
print("Computation time = "+str(1000*(toc - tic ))+"ms") 

n_tic = time.process_time() 
nmultiply = multiply(X, Y) 
n_toc = time.process_time() 
print("Efficient Element wise Product =\n "+ str(nmultiply)); 
print("Computation time = "+str(1000*(n_toc - n_tic ))+"ms") 


# Question 4

def inefficient_sumproduct(x, y):
    """
    Inefficiently sum over all the dimensions of the outer product 
    of two vectors.

    Parameters: 
    x (numpy.ndarray): 1-dimensional numpy array.
    y (numpy.ndarray): 1-dimensional numpy array.

    Returns: 
    numpy.int64: scalar quantity.
    """
    assert(len(x) == len(y))
    
    result = 0
    for i in range(len(x)):
        for j in range(len(y)):
            result += x[i] * y[j]
            
    return result

def sumproduct(x, y):
    """
    Sum over all the dimensions of the outer product of two vectors.

    Parameters: 
    x (numpy.ndarray): 1-dimensional numpy array.
    y (numpy.ndarray): 1-dimensional numpy array.

    Returns: 
    numpy.int64: scalar quantity.
    """

    return sum(x,y)

     
np.random.seed(0)
X = np.random.randint(-1000, 1000, size=3000)
Y = np.random.randint(-1000, 1000, size=3000)

tic = time.process_time() 
isumproduct = inefficient_sumproduct(X, Y) 
toc = time.process_time() 
print("Inefficient Sum Product = "+ str(isumproduct)); 
print("Computation time = "+str(1000*(toc - tic ))+"ms") 

n_tic = time.process_time() 
nsumproduct = sumproduct(X, Y) 
n_toc = time.process_time() 
print("Efficient Sum Product = "+ str(nsumproduct)); 
print("Computation time = "+str(1000*(n_toc - n_tic ))+"ms") 


# Question 5

def inefficient_ReLU(x):
    """
    Inefficiently applies the rectified linear unit function 
    element-wise.

    Parameters: 
    x (numpy.ndarray): 2-dimensional numpy array.

    Returns: 
    numpy.ndarray: 2-dimensional numpy array.
    """
    result = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                result[i][j] = 0
                
    return result

def ReLU(x):
    """
    Applies the rectified linear unit function element-wise.

    Parameters: 
    x (numpy.ndarray): 2-dimensional numpy array.

    Returns: 
    numpy.ndarray: 2-dimensional numpy array.
    """
    

    return np.maximum(x, 0)

np.random.seed(0)
X = np.random.randint(-1000, 1000, size=(3000,3000))


tic = time.process_time() 
irelu = inefficient_ReLU(X) 
toc = time.process_time() 
print("Inefficient ReLu =\n "+ str(irelu)); 
print("Computation time = "+str(1000*(toc - tic ))+"ms") 

n_tic = time.process_time() 
nrelu = ReLU(X) 
n_toc = time.process_time() 
print("Efficient ReLu =\n "+ str(nrelu)); 
print("Computation time = "+str(1000*(n_toc - n_tic ))+"ms") 


# Question 6

def inefficient_PrimeReLU(x):
    """
    Inefficiently applies the derivative of the rectified linear unit 
    function element-wise.

    Parameters: 
    x (numpy.ndarray): 2-dimensional numpy array.

    Returns: 
    numpy.ndarray: 2-dimensional numpy array.
    """

    result = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                result[i][j] = 0
            else:
                result[i][j] = 1
                
    return result

def PrimeReLU(x):
    """
    Applies the derivative of the rectified linear unit function 
    element-wise.

    Parameters: 
    x (numpy.ndarray): 2-dimensional numpy array.

    Returns: 
    numpy.ndarray: 2-dimensional numpy array.
    """
    
    return (x>0).astype(x.dtype) 

np.random.seed(0)
X = np.random.randint(-1000, 1000, size=(3000,3000))


tic = time.process_time() 
iprelu = inefficient_PrimeReLU(X) 
toc = time.process_time() 
print("Inefficient PrimeReLU =\n "+ str(iprelu)); 
print("Computation time = "+str(1000*(toc - tic ))+"ms") 

n_tic = time.process_time() 
prelu = PrimeReLU(X) 
n_toc = time.process_time() 
print("Efficient PrimeReLU =\n "+ str(prelu)); 
print("Computation time = "+str(1000*(n_toc - n_tic ))+"ms") 
print(PrimeReLU(X))


# Section 3

def get_data_1():
    """
    This is the generating process from which example data 1 will derive
    
    Parameters: 
    None
    
    Returns: 
    numpy.ndarray: 1-d numpy array with 2-d numpy arrays as elements.
    """
    freq000 = 3; freq001 = 1; freq002 = 4; freq003 = 1
    freq010 = 5; freq011 = 9; freq012 = 2; freq013 = 6
    freq020 = 5; freq021 = 3; freq022 = 5; freq023 = 8
    frame00 = np.array([freq000, freq001, freq002, freq003])
    frame01 = np.array([freq010, freq011, freq012, freq013])
    frame02 = np.array([freq020, freq021, freq022, freq023])
    utterance0 = np.array([frame00, frame01, frame02])

    freq100 = 9; freq101 = 7; freq102 = 9; freq103 = 3
    freq110 = 2; freq111 = 3; freq112 = 8; freq113 = 4
    frame10 = np.array([freq100, freq101, freq102, freq103])
    frame11 = np.array([freq110, freq111, freq112, freq113])
    utterance1 = np.array([frame10, frame11])

    freq200 = 6; freq201 = 2; freq202 = 6; freq203 = 4
    freq210 = 3; freq211 = 3; freq212 = 8; freq213 = 3
    freq220 = 2; freq221 = 7; freq222 = 9; freq223 = 5
    freq230 = 0; freq231 = 2; freq232 = 8; freq233 = 8
    frame20 = np.array([freq200, freq201, freq202, freq203])
    frame21 = np.array([freq210, freq211, freq212, freq213])
    frame22 = np.array([freq220, freq221, freq222, freq223])
    frame23 = np.array([freq230, freq231, freq232, freq233])
    utterance2 = np.array([frame20, frame21, frame22, frame23])

    spectrograms = np.array([utterance0, utterance1, utterance2])

    return spectrograms

def get_data_2():
    """
    This is the generating process from which example data 2 will derive
    
    Parameters: 
    None
    
    Returns: 
    numpy.ndarray: 1-d numpy array with 2-d numpy arrays as elements.
    """
    np.random.seed(0)
    recordings = np.random.randint(10)
    durations = [np.random.randint(low=5, high=10) 
                 for i in range(recordings)]
    
    data = []
    k = 40 # Given as fixed constant
    for duration in durations:      
        data.append(np.random.randint(10, size=(duration, k)))
    data = np.asarray(data)
    return data


# Question 1

def slice_last_point(x, m):
    """
    Takes one 3-dimensional array with the length of the output instances.
    Your task is to keep only the last m points for each instances in 
    the dataset.

    Parameters: 
    x (numpy.ndarray): 1-d numpy array with 2-d numpy arrays as elements (n, ?, k). 
    m (int): The cutoff reference index in dimension 2.
  
    Returns: 
    numpy.ndarray: A 3-dimensional numpy array of shape (n, m, k)
    """
    spectrograms = x
    
    # Input function dimension specification
    assert(spectrograms.ndim == 1)
    for utter in spectrograms:
        assert(utter.ndim == 2)

    # Pre-define output function dimension specification
    dim1 = spectrograms.shape[0]    # n
    dim2 = m                       # m
    dim3 = spectrograms[0].shape[1] # k

    result = np.zeros((dim1,dim2,dim3))
    
    #### Start of your code ####
    
    result = np.array([row[-m:] for row in spectrograms[:]])


    ####  End of your code  ####

    # Assert output function dimension specification
    assert(result.shape[0] == dim1)
    assert(result.shape[1] == dim2)
    assert(result.shape[2] == dim3)
    
    return result

spectrograms = get_data_1()
duration = 2
print(slice_last_point(spectrograms, duration))

data = get_data_2()
m = 5
print(slice_last_point(data, m)[1])


# Question 2

def slice_fixed_point(x, s, m):
    """
    Takes one 3-dimensional array with the starting position and the 
    length of the output instances. Your task is to slice the instances 
    from the same starting position for the given length.

    Parameters:
    x (numpy.ndarray): 1-d numpy array with 2-d numpy arrays as elements (n, ?, k).
    s (int): The starting reference index in dimension 2.
    m (int): The cutoff reference index in dimension 2.
    
    Returns:
    numpy.ndarray: A 3-dimensional int numpy array of shape (n, m-s, k)
    """
    spectrograms = x
    
    # Input function dimension specification
    assert(spectrograms.ndim == 1)
    for utter in spectrograms:
        assert(utter.ndim == 2)

    # Pre-define output function dimension specification
    dim1 = spectrograms.shape[0]     # n
    dim2 = m-s                      # m-s
    dim3 = spectrograms[0].shape[1]  # k

    result = np.zeros((dim1,dim2,dim3))

    #### Start of your code ####
    
    result = np.array([row[s:m] for row in spectrograms[:]])
    
    ####  End of your code  ####

    # Assert output function dimension specification
    assert(result.shape[0] == dim1)
    assert(result.shape[1] == dim2)
    assert(result.shape[2] == dim3)
    
    return result

spectrograms = get_data_1()
start = 0
end = 2
print(slice_fixed_point(spectrograms, start, end))

data = get_data_2()
s = 2
m = 5
print(slice_fixed_point(data, s, m)[1])


## Question 3

# def slice_random_point(x, d):
#     """
#     Takes one 3-dimensional array with the length of the output instances.
#     Your task is to slice the instances from a random point in each of the
#     utterances with the given length. Please use offset and refer to their 
#     mathematical correspondance.

#     Parameters: 
#     x (numpy.ndarray): 1-d numpy array with 2-d numpy arrays as elements (n, ?, k).
#     d (int): The resulting size of the data in dimension 2.
    
#     Returns: 
#     numpy.ndarray: A 3-dimensional int numpy array of shape (n, d, k)
#     """
#     spectrograms = x
    
#     # Input function dimension specification
#     assert(spectrograms.ndim == 1)
#     for utter in spectrograms:
#         assert(utter.ndim == 2)
#         assert(utter.shape[0] >= d)

#     offset = [np.random.randint(utter.shape[0]-d+1)
#               if utter.shape[0]-d > 0 else 0
#               for utter in spectrograms]

#     # Pre-define output function dimension specification
#     dim1 = spectrograms.shape[0]    # n
#     dim2 = d                       # d
#     dim3 = spectrograms[0].shape[1] # k

#     result = np.zeros((dim1,dim2,dim3))

#     #### Start of your code ####

#     result = np.array([row[np.random.randint(0, d):] for row in spectrograms[:]])
    
    
#     ####  End of your code  ####

#     # Assert output function dimension specification
#     assert(result.shape[0] == dim1)
#     assert(result.shape[1] == dim2)
#     assert(result.shape[2] == dim3)
    
#     return result

# np.random.seed(1)
# spectrograms = get_data_1()
# duration = 2
# print(slice_random_point(spectrograms, duration))


# Question 4

def pad_ending_pattern(x):
    """
    Takes one 3-dimensional array. Your task is to pad the instances from 
    the end position as shown in the example below. That is, you need to 
    pad it with the reflection of the vector mirrored along the edge of the array.
    
    Parameters: 
    x (numpy.ndarray): 1-d numpy array with 2-d numpy arrays as elements.
    
    Returns: 
    numpy.ndarray: 3-dimensional int numpy array
    """
    spectrograms = x
    
    # Input function dimension specification
    assert(spectrograms.ndim == 1)
    for utter in spectrograms:
        assert(utter.ndim == 2)

    # Pre-define output function dimension specification
    dim1 = spectrograms.shape[0]    # n
    dim2 = max([utter.shape[0] for utter in spectrograms]) # m
    dim3 = spectrograms[0].shape[1] # k

    result = np.zeros((dim1, dim2, dim3))

    #### Start of your code ####
    
    list = []
    max_length = max(len(rows) for rows in x)
    for i in x:
        a = np.pad(i, ((0, max_length - len(i)), (0, 0)), "symmetric")
        list.append(a)
    result = np.array(list)
    
    ####  End of your code  ####

    # Assert output function dimension specification
    assert(result.shape[0] == dim1)
    assert(result.shape[1] == dim2)
    assert(result.shape[2] == dim3)
    
    return result

spectrograms = get_data_1()
print(pad_ending_pattern(spectrograms))

# Question 5

def pad_constant_central_pattern(x, cval):
    """
    Takes one 3-dimensional array with the constant value of padding. 
    Your task is to pad the instances with the given constant value while
    maintaining the array at the center of the padding.

    Parameters: 
    x (numpy.ndarray): 1-d numpy array with 2-d numpy arrays as elements.
    cval (numpy.int64): scalar quantity.
    
    Returns: 
    numpy.ndarray: 3-dimensional int numpy array, (n, m, k).
    """
    spectrograms = x
    
    # Input function dimension specification
    assert(spectrograms.ndim == 1)
    for utter in spectrograms:
        assert(utter.ndim == 2)

    dim1 = spectrograms.shape[0]    # n
    dim2 = max([utter.shape[0] for utter in spectrograms]) # m
    dim3 = spectrograms[0].shape[1] # k

    result = np.ones((dim1,dim2,dim3))

    #### Start of your code ####

    list = []
    max_length = max(len(rows) for rows in x)
    for i in x:
        end = int(np.ceil((max_length - len(i)) / 2))
        start = (max_length - len(i)) - end
        a = np.pad(i, ((start, end), (0, 0)), "constant", constant_values=cval)
        list.append(a)
    result = np.array(list)
    
    ####  End of your code  ####

    # Assert output function dimension specification
    assert(result.shape[0] == dim1)
    assert(result.shape[1] == dim2)
    assert(result.shape[2] == dim3)
    
    return result

# Section 4

# Question 1

def tensor_sumproducts(x,y):
    """
    Sum over all the dimensions of the outer product of two vectors.

    Parameters: 
    x (torch.Tensor): 1-dimensional torch tensor.
    y (torch.Tensor): 1-dimensional torch tensor.

    Returns: 
    torch.int64: scalar quantity.
    """
   
    return torch.sum(torch.outer(x,y))

np.random.seed(0)
X = np.random.randint(-1000, 1000, size=3000)
X = torch.from_numpy(X)
Y = np.random.randint(-1000, 1000, size=3000)
Y = torch.from_numpy(Y)

print(tensor_sumproducts(X,Y))


# Question 4

def tensor_ReLU(x):
    """
    Applies the rectified linear unit function element-wise.

    Parameters: 
    x (torch.Tensor): 2-dimensional torch tensor.

    Returns: 
    torch.Tensor: 2-dimensional torch tensor.
    """
    
    
    return torch.clamp(x, min=0)

np.random.seed(0)
X = np.random.randint(-1000, 1000, size=(1000,1000))
X = torch.from_numpy(X)

print(tensor_ReLU(X))


# Question 5

def tensor_ReLU_prime(x):
    """
    Applies derivative of the rectified linear unit function 
    element-wise.

    Parameters: 
    x (torch.Tensor): 2-dimensional torch tensor.

    Returns: 
    torch.Tensor: 2-dimensional torch tensor.
    """
   
    x = x.numpy()
    x[x >= 0] = 1
    x[x < 0] = 0
    return torch.from_numpy(x) 

np.random.seed(0)
X = np.random.randint(-1000, 1000, size=(1000,1000))
X = torch.from_numpy(X)

print(tensor_ReLU_prime(X))