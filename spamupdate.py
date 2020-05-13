import numpy as np

def spamupdate(w,email,truth):

    # Input:
    # w     weight vector
    # email instance vector
    # truth label
    #
    # Output:
    #
    # updated weight vector
    #
    # INSERT CODE HERE:
    pred = np.sign(np.matmul(w.T,email))
    
    
    if pred != truth:
        temp = np.matmul(email.reshape(np.shape(w)[0],1),truth.reshape(1,1))
        w = w + temp

    return w
