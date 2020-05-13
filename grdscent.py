
from scipy.linalg import norm
from numpy import maximum

def grdescent(func,w0,stepsize,maxiter,tolerance=0.1):
# INPUT:
# func function to minimize
# w_trained = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUTS:
#
# w = final weight vector
    eps = 2.2204e-14 #minimum step size for gradient descent
    w = w0
    
    loss,gradient = func(w0)
    
    for i in range(maxiter):
        updated_loss,gradient = func(w)
        
        # check whether norm of gradient is larger than tolerance
        if (norm(gradient) >= tolerance):
            # Update stepsize through comparing loss values
            if(updated_loss < loss):
                stepsize = stepsize * 1.01
            else:
                stepsize = stepsize * 0.5
                # check whether step size is larger than minimum step size
                stepsize = maximum(stepsize, eps)
            
            
            # Update weights
            w = w - stepsize * gradient
        else:
            break
        
        loss = updated_loss

  
    return w
