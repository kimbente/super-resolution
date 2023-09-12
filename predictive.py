import torch
import numpy as np
import math

########################################
### CHOLESKY inverse with correction ###
########################################

def predictive_distribution_cholesky_correction(lr_variable_channel, k_ah_ah, k_ah_al, k_al_al, noise = 0.05, mu = 0.5, batch_size = 1):
    """Calculation of the mean and variance of the Gaussian predictive distribution as well as of Log Marginal Likelihood. 
    Here we implement the inversion through the Cholesky decomposition [torch.linalg.cholesky()] followed by the inversion [torch.cholesky_inverse(L)]. 
    This is slighly different to the Algorithm 2.10 proposed in Rasmussen & Williams but more suitable to implement the weight correction performed by Reid et al.

    Args:
        lr_variable_channel (_type_): _description_
        k_ah_ah (_type_): _description_
        k_ah_al (_type_): _description_
        k_al_al (_type_): _description_
        noise (float, optional): _description_. Defaults to 0.05.
        mu (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    # Calculate inverse via the Cholesky decomposition (Lower Triangular)
    L = torch.linalg.cholesky(k_al_al + (torch.eye(n = k_al_al.shape[-1]) * noise))
    k_inv = torch.cholesky_inverse(L)
    
    #### Mean ###
    pl_al_minus_mu = lr_variable_channel - torch.ones(size = lr_variable_channel.shape) * mu
    W = torch.matmul(k_ah_al, k_inv) 

    # Correction
    mean_flat = torch.div(torch.matmul(W, pl_al_minus_mu.reshape(-1).unsqueeze(1)), torch.matmul(W, torch.ones(size = (W.shape[-1], 1)))) + mu
    # cast into 2D shape
    mean = mean_flat.reshape(batch_size, int(np.sqrt(mean_flat.shape[0])), -1)

    ### Sigma ###
    # .mT transposes the last two dims of a matrix
    sigma = k_ah_ah - torch.matmul(k_ah_al, torch.matmul(k_inv, k_ah_al.mT))

    ### LML ###
    # 2.30 in Rasmussen
    # https://d2l.ai/chapter_gaussian-processes/gp-inference.html
    
    # Term1: Kernel term
    term1 = torch.mul(torch.matmul(lr_variable_channel.mT.reshape(1, -1), torch.matmul(k_inv, lr_variable_channel.reshape(-1, 1))), 0.5)
    
    # Term2: Determinant term. 2 and 0.5 cancel each other out, sum in log space, log makes values negative
    term2 = torch.sum(torch.log(torch.diagonal(L)))
    # Need trick https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
    # Does not work: term2 = torch.mul(torch.log(torch.linalg.det(k_al_al + (torch.eye(n = k_al_al.shape[-1]) * noise))), 0.5).reshape(1, 1)
    # Flatten shape and extract n, natural log
    
    # Term3: Constant term
    term3 = (torch.log(torch.tensor(2 * math.pi)) * torch.tensor(lr_variable_channel.reshape(-1).shape[0] * 0.5)).reshape((1, 1)) # n
    lml = - term1 - term2 - term3

    return mean, sigma, lml


#####################################
### CHOLESKY solve, no correction ###
#####################################

# see Rasmussen & Williams Algorithm 2.1 

def predictive_distribution_cholesky(lr_variable_channel, k_ah_ah, k_ah_al, k_al_al, noise = 0.05, mu = 0.5):
    """Combine mean and variance generation into one.
    Following the algorithm in Rasmussen 2.1 and https://gregorygundersen.com/blog/2019/09/12/practical-gp-regression/
    no correction

    Args:
        lr_variable_channel (_type_): _description_
        k_ah_ah (_type_): _description_
        k_ah_al (_type_): _description_
        k_al_al (_type_): _description_
        noise (float, optional): _description_. Defaults to 0.05.
        mu (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    # Lower triangular Cholesky decomposition
    L = torch.linalg.cholesky(k_al_al + (torch.eye(n = k_al_al.shape[-1]) * noise)).unsqueeze(0)
    
    #### Mean ###
    pl_al_minus_mu = lr_variable_channel - torch.ones(size = lr_variable_channel.shape) * mu

    # debug

    # https://pytorch.org/docs/stable/generated/torch.cholesky_solve.html
    # Solve linear system instead of inversion: 
    # Input1: torch.Size([1, 900, 1]), Input2: torch.Size([1, 900, 900])
    alpha = torch.cholesky_solve(pl_al_minus_mu.reshape(-1).unsqueeze(1).unsqueeze(0), L, upper = False)
    # alpha shape torch.Size([1, 900, 1])

    print(k_ah_al.shape)
    print(alpha.shape)

    # No correction
    mean_flat = torch.matmul(k_ah_al.unsqueeze(0), alpha) + mu
    # cast into 2D shape
    batch_size = 1
    mean = mean_flat.reshape(batch_size, int(np.sqrt(mean_flat.shape[1])), -1)

    print(mean_flat.shape)
    print(mean.shape)

    ### Sigma ###
    v = torch.cholesky_solve(k_ah_al.mT, L, upper = False)
    # .mT transposes the last two dims of a matrix
    sigma = k_ah_ah - torch.matmul(k_ah_al, v)

    ### LML ###
    # 2.30 in Rasmussen
    # https://d2l.ai/chapter_gaussian-processes/gp-inference.html
    
    # Term1: Kernel term
    term1 = torch.mul(torch.matmul(lr_variable_channel.mT.reshape(1, -1), alpha), 0.5)
    
    # Term2: Determinant term. 2 and 0.5 cancel each other out, sum in log space, log makes values negative
    term2 = torch.sum(torch.log(torch.diagonal(L)))
    # Need trick https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
    # Does not work: term2 = torch.mul(torch.log(torch.linalg.det(k_al_al + (torch.eye(n = k_al_al.shape[-1]) * noise))), 0.5).reshape(1, 1)
    # Flatten shape and extract n, natural log
    
    # Term3: Constant term
    term3 = torch.tensor(torch.log(torch.tensor(2 * math.pi)) * torch.tensor(lr_variable_channel.reshape(-1).shape[0] * 0.5)).reshape((1, 1)) # n
    lml = - term1 - term2 - term3

    return mean, sigma, lml

######################################
### linalg inverse with correction ###
######################################

def predictive_distribution(lr_variable_channel, k_ah_ah, k_ah_al, k_al_al, noise = 0.05, mu = 0.5):
    """Combine mean and variance generation into one

    Args:
        lr_variable_channel (_type_): _description_
        k_ah_ah (_type_): _description_
        k_ah_al (_type_): _description_
        k_al_al (_type_): _description_
        noise (float, optional): _description_. Defaults to 0.05.
        mu (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    # Calculate inverse once. Replace with cholesky
    k_inv = torch.linalg.inv(k_al_al + (torch.eye(n = k_al_al.shape[-1]) * noise))
    
    #### Mean ###
    pl_al_minus_mu = lr_variable_channel - torch.ones(size = lr_variable_channel.shape) * mu
    W = torch.matmul(k_ah_al, k_inv) 
    mean_flat = torch.div(torch.matmul(W, pl_al_minus_mu.reshape(-1).unsqueeze(1)), torch.matmul(W, torch.ones(size = (W.shape[-1], 1)))) + mu
    # cast into 2D shape
    mean = mean_flat.reshape(int(np.sqrt(mean_flat.shape[0])), -1)

    ### Sigma ###
    # .mT transposes the last two dims of a matrix
    sigma = k_ah_ah - torch.matmul(k_ah_al, torch.matmul(k_inv, k_ah_al.mT))

    return mean, sigma