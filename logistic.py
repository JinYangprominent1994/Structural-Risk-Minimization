import numpy as np

'''

    INPUT:
    xTr dxn matrix (each column is an input vector)
    yTr 1xn matrix (each entry is a label)
    w weight vector (default w=0)

    OUTPUTS:

    loss = the total loss obtained with w on xTr and yTr
    gradient = the gradient at w

    [d,n]=size(xTr);
'''
def logistic(w,xTr,yTr):

    # YOUR CODE HERE
    
    # Calculate loss
    loss = np.sum(np.log(1 + np.exp(-np.matmul(w.T, xTr) * yTr)))
    
    # Calculate gradient
    gradient = 0 - np.matmul(xTr, (1/(1 + np.exp(np.matmul(w.T, xTr) * yTr)) * yTr).T)
    
    return loss,gradient
