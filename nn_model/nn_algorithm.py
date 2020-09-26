import numpy as np

#initalize parameters
#layer_dims = katmanların nöron sayılarını tutan liste (özellikler dahil)
def initilaize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L): 
        #np.sqrt(layer_dims[l-1]) sayesinde W parametresini daha küçük sayılara indirgiyoruz ve öğrenimini arttırıyoruz. 0.01 gibi sayılarlada çarpabiliriz.
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) / np.sqrt(layer_dims[l-1])# W(l,l-1)
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1)) # b(l,1)
    return parameters

def linear_forward(A_prev,W,b):
    Z = np.dot(W,A_prev) + b # Z = WA + b (vectorized)
    assert(Z.shape == (W.shape[0],A_prev.shape[1]))
    cache = (A_prev,W,b)
    return Z, cache

def sigmoid(Z): #activation function
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache # Eğer relu kullanmazsanız Z yerine A yı cache'e atın. 

def relu(Z): #activation function
    A = np.maximum(0,Z)
    cache = Z
    return A, cache

def linear_activation_forward(A_prev,W,b,activation):
    Z, linear_cache = linear_forward(A_prev,W,b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    cache = (linear_cache,activation_cache) #backpropagation için gerekli değerler
    return A,cache

def nn_forward_propagation(X,parameters): #Sınıflandırma problemleri için tasarlanmıştır.
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1,L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,parameters['W' + str(l)],parameters['b' + str(l)], activation="relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A,parameters['W' + str(L)],parameters['b' + str(L)], activation="sigmoid")
    caches.append(cache)
    assert(AL.shape == (1,X.shape[1]))
    return AL, caches

def cost_function(AL,Y): #tahmindeki hatayı gösterir.
    m = Y.shape[1]

    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y,np.log(1-AL).T))
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost

def linear_backward(dZ,cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1./m) * np.dot(dZ,A_prev.T)
    db = (1./m) * np.sum(dZ,axis=1,keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    return dA_prev, dW, db

def sigmoid_backward(dA,cache):
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def relu_backward(dA,cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert(dZ.shape == Z.shape)
    return dZ

def linear_activation_backward(dA,cache,activation): 
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)
        dA_prew, dW, db = linear_backward(dZ,linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA,activation_cache)
        dA_prew, dW, db = linear_backward(dZ,linear_cache)
    return dA_prew, dW, db 

def nn_backward_propagation(AL,Y,caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y,AL) - np.divide(1-Y,1-AL)) #Cost function türevi

    current_cache = caches[L - 1]
    grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL,current_cache,activation="sigmoid")

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        grads['dA' + str(l)], grads['dW' + str(l+1)], grads['db' + str(l+1)] = linear_activation_backward(grads['dA'+str(l+1)],current_cache,activation="relu") 
    return grads

def update_parameters(parameters,grads,learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return parameters

def predict(X,parameters):
    AL, cache = nn_forward_propagation(X,parameters)
    predictions = (AL>0.5)
    return predictions

def accuracy(predict,Y):
    accury = np.squeeze(((np.dot(Y,predict.T) + np.dot(1-Y,1-predict.T))/float(Y.size)) * 100)
    return accury