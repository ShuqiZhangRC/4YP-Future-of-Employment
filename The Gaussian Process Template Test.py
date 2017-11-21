# -*- coding: utf-8 -*-
import numpy as np
#import numpy and treat numpy as np in this code,
#Numpy is the core library for scientific computing 
#in Python. It provides a high-performance 
#multidimensional array object, and tools for working
#with these arrays. 

def exponential_cov(x, f, params):
    '''this function returns a covariance function, the parameters are 
    hyperparameters for the calculation of mean and variance, this 
    function can be changed according to the features of the input data'''
    return params[0] * np.exp( -0.5 * params[1] * np.subtract.outer(x, f)**2)
#np.subtract.outer(x,y) - the function of subrtaction 
#is applied to all pairs of (x,y)

def conditional(x_new, x, f, params):
    '''the function for the conditonal mean and
     covariance '''
    B = exponential_cov(x_new, x, params)
    #the covariance of x_new and x with the hyperparamters, K_new
    C = exponential_cov(x, x, params)
    #the covariance of x and x from the training data, K
    A = exponential_cov(x_new, x_new, params)
    #the covaraince of x_new and x_new from the test data. K_newnew

    #as we want to form the covaraince matrix of [[C] B; B.T A],
    #therefore, the mean and the covariance for the conditional 
    #distribution is mu and sigma.
    

    mu = np.linalg.inv(C).dot(B.T).T.dot(f)
    #mu is calcualted by a linear algebra tool.
    #np.linalg.inv(C) returns the inverse of matrix C
    #mu = mu(x_new) + K_new.T * K^-1 * (f - mu(x)) 
    #in this case, the mu for the training data and the test data
    #are assumed to be zero. so mu = K_new.T * K^-1 * f

    sigma = A - (B.T).dot(np.linalg.inv(C).dot(B))
    #sigma = K_newnew - K_new.T * K^-1 * K_new 

    return(mu.squeeze(), sigma.squeeze())
    #the dimensions of mu and sigma are decreased by using projection 
    #with the code squeeze()
    
import matplotlib.pylab as plt
#the gaussian prior is the function for the mean 
#and the variance using the hyperparameters
theta = [1, 5]
sigma_0 = exponential_cov(0, 0, theta)
#the covaraince for no training data

xpts = np.arange(-3, 3, step=0.01)
#it's pure error function withou any training data




#plt.errorbar(xpts, np.zeros(len(xpts)), yerr=sigma_0, capsize=0)
#plt.show()




#x = [1.]
#f = [np.random.normal(scale=sigma_0)]

x = [1, 2, 3]
f = [2, 5, 7]

sigma_1 = exponential_cov(x, x, theta)


def predict(data_test, data_training_x, kernel, params, sigma, data_training_f):
	

    k = [kernel(data_test, x, params) for x in data_training_x]
	#kernel is the function returns the covariance
	
    CovInv = np.linalg.inv(sigma)
    #sigma is K

    f_pred = np.dot(k, CovInv).dot(data_training_f)
    #the mean gives f_pred = K_new * K^-1 * f

    sigma_new = kernel(data_test, data_test, params) - np.dot(k, CovInv).dot(k)
    #covariance is K_newnew - K_new.T * K^-1 * k_new

    return f_pred, sigma_new

x_pred = np.linspace(-3, 3, 1000)
predictions = [predict(i, x, exponential_cov, theta, sigma_1, f) for i in x_pred]



print(predictions)
#preditions give data in form like (f_pred, sigma_new) (f_pred, sigma_new)

f_pred, sigmas = np.transpose(predictions)
#transpose the preditions into (f_pred, f_pred) and (sigma_new, sigma_new)

plt.errorbar(x_pred, f_pred, yerr=sigmas, capsize=0)
#each f_pred is plotted against x_pred which is just 
#	a linspace of input data with errorbar size = sigmas
#	in y direction.

#Plot x versus y with error deltas in yerr and xerr.
#	Vertical errorbars are plotted if yerr is not None.
#   Horizontal errorbars are plotted if xerr is not None.
#    
#   x, y, xerr, and yerr can all be scalars, which plots a
#   single error bar at x, y.

print(f_pred, sigmas)

plt.plot(x, f, "ro")

plt.show()


