from spamfilter import spamfilter
from trainspamfilter import trainspamfilter
from valsplit import valsplit
from scipy import io

#import numpy as np


#from ridge import ridge
#from hinge import hinge
#from logistic import logistic
#from checkgradLogistic import checkgradLogistic
#from checkgradHingeAndRidge import checkgradHingeAndRidge

# load the data:
data = io.loadmat('data/data_train_default.mat')
X = data['X']
Y = data['Y']

# split the data:
xTr,xTv,yTr,yTv = valsplit(X,Y)

# train spam filter with settings and parameters in trainspamfilter.py
w_trained = trainspamfilter(xTr,yTr)

# evaluate spam filter on test set using default threshold
spamfilter(xTv,yTv,w_trained)