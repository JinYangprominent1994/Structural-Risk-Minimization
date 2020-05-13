
import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    # YOUR CODE HERE
       
    # Calculate the loss function
    loss = np.matmul((np.matmul(w.T, xTr) - yTr),(np.matmul(w.T, xTr) - yTr).T) + lambdaa * np.matmul(w.T, w)
    
    # Calculate the gradient function
    gradient = 2 * np.matmul(xTr,(np.matmul(w.T, xTr) - yTr).T) + 2 * lambdaa * w 

    return loss,gradient
