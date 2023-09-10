import torch
import numpy as np
import math

###############
### WRAPPER ###
###############

def covariance_function(kp_ds, lambda_s = 0.3, lambda_p = 0.4, sigma_f = 1.0):
    """_summary_

    Args:
        kp_ds (_type_): _description_
        lambda_s (float, optional): _description_. Defaults to 0.3.
        lambda_p (float, optional): _description_. Defaults to 0.4.
        sigma_f (float, optional): _description_. Defaults to 1.0.

    Returns:
        torch.tensor: returns covariance matrix
    """

    # Assuming square shape: H == W
    dims = kp_ds.shape[1]

    ### Mid-points norm ###
    # Create input for ks kernel: mid_points_norm
    # arange function exclude the last one: (1, 46) goes up to (not including) 46
    xs = torch.arange(1, (dims + 1)).repeat(dims, 1) # increasing in x direction
    ys = xs.T # increasing in y direction
    # concat y and x midpoint channel and normalise so they range from 0 to 1
    mid_points_norm = torch.cat((ys.unsqueeze(0), xs.unsqueeze(0)), dim = 0)/dims

    ### Ks ### smooth, sparse, local
    # lambda_s is the hp that controls the receptive field
    ks_input = mid_points_norm[:, :, :]
    ks_covar_matrix = ks_covariance_function(tensor = ks_input, lambda_value = lambda_s)

    ### Kp ### coupling(decoupling) of similar(dissimilar) pixel values, non-stationary
    # lambda_p is the lengthscale of the RBF kernel
    # default 0.6931
    kp_input = kp_ds[:, :]
    kp_covar_matrix = kp_covariance_function(kp_input, lambda_p = lambda_p)
    
    # Return components as well
    # return torch.mul(torch.matmul(ks_covar_matrix, kp_covar_matrix), sigma_f)
    return torch.tensor((ks_covar_matrix.detach().numpy() * kp_covar_matrix.detach().numpy() * sigma_f))

#########################
### Spatial component ###
#########################

def ks_covariance_function(tensor, lambda_value = 0.2):
    """ spatial covariance function

    Args:
        tensor (torch.tensor): _description_
        lambda_s (float, optional): _description_. Defaults to 0.2.

    Returns:
        torch.tensor: spatial covariance 
    """
    # If tensor is not flat, flatten
    if len(np.array(tensor.shape)) > 2:
        # 2 is hardcoded
        tensor = tensor.reshape(2, -1)

    ### Euclidean distance (2D) ###
    # broadcast and calculate pairwise distances 
    dist = tensor.unsqueeze(-1) - tensor.unsqueeze(-2)
    dist_sqr = torch.pow(dist, exponent = 2)
    # sum across x and y axis
    dist_sum = torch.sum(dist_sqr, dim = 0)
    # take sqrt
    euc_dist = torch.sqrt(dist_sum)
    
    Z = torch.div(euc_dist, lambda_value)

    # Mask large Z's with nan before replacing values with zero
    Z[Z >= 1] = float('nan')

    # first term pushes small distanced to 0 and distances near 1 close to zero
    # second terms is clipped at 1 so that small distances will approach 1
    cov_matrix = torch.pow((1 - Z), exponent = 3) * ((3 * Z) + 1)
    cov_matrix[torch.isnan(cov_matrix)] = 0.0

    return cov_matrix

#######################
### Pixel component ###
#######################

# Write my own 1D RBF
def kp_covariance_function(tensor, lambda_p = 0.6):
    """ calculate pixel-intensity RBF covariance function.

    Args:
        tensor (torch.tensor): 2D or 1D tensor but with only one variable
        lambda_p (float, optional): Defaults to 0.6.

    Returns:
        _type_: rbf covariance 
    """
    # Assuming 1D
    # If tensor is not flat, flatten
    # Edit: use np.array() instead of list()
    if len(np.array(tensor.shape)) > 1:
        tensor = tensor.reshape(-1)
    # broadcast shape to calculate pariwise distances
    dist = tensor.unsqueeze(-1) - tensor.unsqueeze(-2)
    # direction of distance does not matter
    dist_sqr = torch.pow(dist, exponent = 2)
    dist_sqr_scaled = dist_sqr/(2 * torch.pow(torch.tensor(lambda_p), exponent = 2))
    # 1 at zero distance
    rbf = torch.exp(- dist_sqr_scaled)

    return rbf      


##############################
### Covariance aggregation ###
##############################

def aggregation_dictionaries(hr_matrix, lr_2D):

    ### Check that hr_matrix (base_covariance matrix) covers sufficient area
    # y_min: rows and columns contain the same so choose row. Could find min based on on position but use torch.min instead
    if ((torch.min(lr_2D[0, :, :]) < torch.min(hr_matrix[1, :, :])) or # min of y_min
        (torch.max(lr_2D[1, :, :]) > torch.max(hr_matrix[2, :, :])) or # max of y_max
        (torch.min(lr_2D[2, :, :]) < torch.min(hr_matrix[3, :, :])) or # min of x_min
        (torch.max(lr_2D[3, :, :]) > torch.max(hr_matrix[4, :, :]))): # max of x_max
        print("We have an issue. The lr area is not covered by the hr area")
    
    # Number of cells spanned by a_l might vary. Thus tensor strucuture not ideal
    # dict for each output covariance value with area (for weighting) and with values

    # Flatten lr shape
    lr = lr_2D.reshape(4, -1)

    # Extract dimensionalities of each for the loop
    hr_dims_flat = np.array(hr_matrix.shape)[-1]
    lr_dims_flat = np.array(lr).shape[-1]

    # Create two dictionaries with empty lists that you can append to
    covariance_index_dict = {(lr_index): [] for lr_index in range(0, lr_dims_flat)}
    weights_dict = {(lr_index): [] for lr_index in range(0, lr_dims_flat)}

    joker = 0

    for i in range(0, lr_dims_flat):
        # Check overlap between lr and "columns" of hr (last 4 channels) of hr_matrix
        # both axis (y and x) need to overlap for there to be an area.
        for j in range(0, hr_dims_flat):
            # First: check y overlap. If hr_y_max < lr_y_min -> no y overlap or hr_y_min > lr_y_max
            # (hr_matrix[:, 0, :] select random row (here 0) since y_max values don't 
            if ((hr_matrix[6, joker, j] <= lr[0, i]) or # hr y_max <= lr y_min
                (hr_matrix[5, joker, j] >= lr[1, i])): # hr y_min >= lr y_max
                # NO y overlap, move to next column.
                j += 1
                # Note: computionally cheaper if we check one first
            elif ((hr_matrix[5, joker, j] < lr[1, i]) & # lr y_min <= hr y_min < lr_y_max
                  (hr_matrix[5, joker, j] >= lr[0, i])):
                # YES y overlap, may be partial or full
                if (hr_matrix[6, joker, j] <= lr[1, i]): # if hr y_max <= lr y_max
                    # YES full y overlap

                    ### x block ###
                    if ((hr_matrix[8, joker, j] <= lr[2, i]) or # hr x_max <= lr x_min
                        (hr_matrix[7, joker, j] >= lr[3, i])): # hr x_min >= lr x_max
                        # NO x overlap, move to next column
                        j += 1
                    elif ((hr_matrix[8, joker, j] > lr[2, i]) &  # hr x_max > lr x_min
                          (hr_matrix[8, joker, j] <= lr[3, i])): # hr x_max <= lr x_max
                        # YES x overlap, may be partial or full
                        if (hr_matrix[7, joker, j] >= lr[2, i]): # hr x_min >= lr x_min
                            # FULL y and FULL x overlap:
                            A = (hr_matrix[6, joker, j] - hr_matrix[5, joker, j]) * (hr_matrix[8, joker, j] - hr_matrix[7, joker, j]) # (hr y_max - hr y_min) * (hr x_max - hr x_min)
                            V = hr_matrix[6, :, j] # Vector of values

                            # Append covariance index to covariance index dictionary
                            covariance_index_dict[i].append(j)
                            # Append Area to weights dict
                            weights_dict[i].append(A)

                            j += 1
                        else:
                            # FULL y and partial x overlap (hr extending leftwards of lr in x direction)
                            A = (hr_matrix[6, joker, j] - hr_matrix[5, joker, j]) * (hr_matrix[8, joker, j] - lr[2, i]) # (hr y_max - hr y_min) * (hr x_max - lr x_min)
                            V = hr_matrix[0, :, j] # Vector of values

                            # Append covariance index to covariance index dictionary
                            covariance_index_dict[i].append(j)
                            # Append Area to weights dict
                            weights_dict[i].append(A)
                        
                            j += 1
                    elif ((hr_matrix[7, joker, j] >= lr[2, i]) & # lr x_min <= hr x_min < lr x_max
                          (hr_matrix[7, joker, j] < lr[3, i])):
                        # FULL y and partial x overlap: (hr extending rightwards of lr in x direction)

                        A = (hr_matrix[6, joker, j] - hr_matrix[5, joker, j]) * (lr[3, i] - hr_matrix[7, joker, j]) # (hr y_max - hr y_min) * (lr x_max - hr x_min)
                        V = hr_matrix[0, :, j] # Vector of values. channel 0, all rows, current column

                        # Append covariance index to covariance index dictionary
                        covariance_index_dict[i].append(j)
                        # Append Area to weights dict
                        weights_dict[i].append(A)

                        j += 1
                    else:
                        print("We didn't catch this case.")
                    ### x block end ###

                else:
                    # YES partial y overlap.
                    
                    ### x block ###
                    if ((hr_matrix[8, joker, j] <= lr[2, i]) or # hr x_max <= lr x_min
                        (hr_matrix[7, joker, j] >= lr[3, i])): # hr x_min >= lr x_max
                        # NO x overlap, move to next column
                        j += 1
                    elif ((hr_matrix[8, joker, j] > lr[2, i]) &  # hr x_max > lr x_min
                          (hr_matrix[8, joker, j] <= lr[3, i])): # hr x_max <= lr x_max
                        # YES x overlap, may be partial or full
                        if (hr_matrix[7, joker, j] >= lr[2, i]): # hr x_min >= lr x_min
                            # Partial y and FULL x overlap (with hr extending upwards of lr)

                            A = (lr[1, i] - hr_matrix[5, joker, j]) * (hr_matrix[8, joker, j] - hr_matrix[7, joker, j]) # (lr y_max - hr y_min) * (hr x_max - hr x_min)
                            V = hr_matrix[0, :, j] # Vector of values. channel 0, all rows, current column

                            # Append covariance index to covariance index dictionary
                            covariance_index_dict[i].append(j)
                            # Append Area to weights dict
                            weights_dict[i].append(A)

                            j += 1
                        else:
                            # Partial y and partial x overlap: (with hr extending upwards of lr in y direction, and hr extending leftwards of lr in x direction)

                            A = (lr[1, i] - hr_matrix[5, joker, j]) * (hr_matrix[8, joker, j] - lr[2, i]) # (lr y_max - hr y_min) * (hr x_max - lr x_min)
                            V = hr_matrix[0, :, j] # Vector of values. channel 0, all rows, current column

                            # Append covariance index to covariance index dictionary
                            covariance_index_dict[i].append(j)
                            # Append Area to weights dict
                            weights_dict[i].append(A)

                            j += 1
                    elif ((hr_matrix[7, joker, j] >= lr[2, i]) & # lr x_min <= hr x_min < lr x_max
                          (hr_matrix[7, joker, j] < lr[3, i])):
                        # Partial y and partial x overlap: (with hr extenting upwards in y direction, with hr extending rightwards of lr in x sirection)

                        A = (lr[1, i] - hr_matrix[5, joker, j]) * (lr[3, i] - hr_matrix[7, joker, j]) # (lr y_max - hr y_min) * (lr x_max - hr x_min)
                        V = hr_matrix[0, :, j] # Vector of values. channel 0, all rows, current column

                        # Append covariance index to covariance index dictionary
                        covariance_index_dict[i].append(j)
                        # Append Area to weights dict
                        weights_dict[i].append(A)
                    
                        j += 1
                    else:
                        print("We didn't catch this case.")
                    ### x block end ###

            elif ((hr_matrix[6, joker, j] > lr[0, i]) & # lr y_min < hr y_max <= lr_y_max
                  (hr_matrix[6, joker, j] <= lr[1, i])):
                  # YES partial y overlap.
                
                    ### x block ###
                    if ((hr_matrix[8, joker, j] <= lr[2, i]) or # hr x_max <= lr x_min
                        (hr_matrix[7, joker, j] >= lr[3, i])): # hr x_min >= lr x_max
                        # NO x overlap, move to next column
                        j += 1
                    elif ((hr_matrix[8, joker, j] > lr[2, i]) &  # hr x_max > lr x_min
                          (hr_matrix[8, joker, j] <= lr[3, i])): # hr x_max <= lr x_max
                        # YES x overlap, may be partial or full
                        if (hr_matrix[7, joker, j] >= lr[2, i]): # hr x_min >= lr x_min
                            # Partial y and FULL x overlap: (with hr extending downwards of lr in y direction)
                            
                            A = (hr_matrix[6, joker, j] - lr[0, i]) * (hr_matrix[8, joker, j] - hr_matrix[7, joker, j]) # (hr y_max - lr y_min) * (hr x_max - hr x_min)
                            V = hr_matrix[0, :, j] # Vector of values. channel 0, all rows, current column

                            # Append covariance index to covariance index dictionary
                            covariance_index_dict[i].append(j)
                            # Append Area to weights dict
                            weights_dict[i].append(A)

                            j += 1

                        else:
                            # Partial y and partial x overlap: (with hr extending downwards of lr in y direction, and with hr extending leftwars in x direction)
                            A = (hr_matrix[6, joker, j] - lr[0, i]) * (hr_matrix[8, joker, j] - lr[2, i]) # (hr y_max - lr y_min) * (hr x_max - lr x_min)
                            V = hr_matrix[0, :, j] # Vector of values. channel 0, all rows, current column

                            # Append covariance index to covariance index dictionary
                            covariance_index_dict[i].append(j)
                            # Append Area to weights dict
                            weights_dict[i].append(A)
                    
                            j += 1
                    elif ((hr_matrix[7, joker, j] >= lr[2, i]) & # lr x_min <= hr x_min < lr x_max
                          (hr_matrix[7, joker, j] < lr[3, i])):
                        # Partial y and partial x overlap: (with hr extending downwards of lr in y direction, with hr extending rightwards of lr in x direction)

                        A = (hr_matrix[6, joker, j] - lr[0, i]) * (lr[3, i] - hr_matrix[7, joker, j]) # (hr y_max - lr y_min) * (lr x_max - hr x_min)
                        V = hr_matrix[0, :, j] # Vector of values. channel 0, all rows, current column

                        # Append covariance index to covariance index dictionary
                        covariance_index_dict[i].append(j)
                        # Append Area to weights dict
                        weights_dict[i].append(A)
                    
                        j += 1
                    else:
                        print("We didn't catch this case.")
                    ### x block end ###

            else:
                print("We didn't catch this case.")
        
    return covariance_index_dict, weights_dict


################

def aggregate_base_covariance_matrix(hr_matrix, lr_2D):
    """Aggregate the base covariance matrix into k_ah_al, and k_al_al. Box-channel representations required (as opposed to mid-points)
    Assuming symmetry.
    aggregation_dictionaries

    Args:
        hr_matrix (torch.tensor): shape [9, hr_dims_flat, hr_dims_flat]. base covariance matrix (high-resolution). 9 channels are 1 + 4 + 4/
        lr_2D (torch.tensor): shape [4, lr_dims, lr_dims]. Low resolution box-channels: y_min, y_max, x_min, x_max for every target grid cell. 

    Returns:
        torch.tensor: k_ah_al_tensor, 
        torch.tensor: k_al_al_tensor
    """

    # Flatten lr shape: box is 4
    lr_flat = lr_2D.reshape(4, -1)

    # Extract dimensionalities of each for the loop
    hr_dims_flat = np.array(hr_matrix.shape)[-1]
    lr_dims_flat = np.array(lr_flat.shape)[-1]

    covariance_indices_dict, weights_dict = aggregation_dictionaries(hr_matrix, lr_2D)
    
    ###############
    ### k_ah_al ###
    ###############

    # Initilise new empty tensor with correct number of rows
    k_ah_al_tensor = torch.empty(size = (hr_dims_flat, 0))

    for i in range(0, lr_dims_flat):

        # Normaliser is the area of the target grid cell (constant for regular grids)
        weight_normaliser = torch.sum(torch.tensor(weights_dict[i], dtype = float))

        # Normlised weights: torch.Size([n, 1]) where n can vary based on the number over overlapping grid cells with target cell
        # weights add up to 1 for every i
        weights = torch.div(torch.tensor(weights_dict[i], dtype = float), weight_normaliser).unsqueeze(-1)

        # Extract columns based on covariance_index_dict to get tensor of shape [hr_dims_flat, n]  
        # where n can vary based on the num over overlapping grid cells
        # Use only first dim of hr_matrix and all rows.
        column_vectors = hr_matrix[0, :, covariance_indices_dict[i]].double()

        aggregated_column_vector = torch.matmul(column_vectors, weights)
        
        # Concat with empty tensor
        k_ah_al_tensor = torch.cat((k_ah_al_tensor, aggregated_column_vector), dim = 1)
    
    ###############
    ### k_al_al ###
    ###############

    # Transpose so shape is [lr_dims_flat, hr_dims_flat]
    k_ah_al_tensor_transpose = torch.transpose(k_ah_al_tensor, dim0 = 1, dim1 = 0)

    # Initilise new empty tensor with correct number of rows
    k_al_al_tensor = torch.empty(size = (lr_dims_flat, 0))

    for i in range(0, lr_dims_flat):
        # Normaliser is the area of the grid cell (constant for regular grids)
        weight_normaliser = torch.sum(torch.tensor(weights_dict[i], dtype = float))
        # Normlised weights: torch.Size([n, 1]) where n can vary based on the number over overlapping grid cells with current cell
        # Add up to 1
        weights = torch.div(torch.tensor(weights_dict[i], dtype = float), weight_normaliser).unsqueeze(-1)

        # Extract columns based on covariance_index_dict to get tensor of shape [lr_dims_flat, n] 
        # where n can vary based on the num over overlapping grid cells
        column_vectors = k_ah_al_tensor_transpose[:, covariance_indices_dict[i]]
        aggregated_column_vector = torch.matmul(column_vectors, weights)

        # Concat with empty tensor
        k_al_al_tensor = torch.cat((k_al_al_tensor, aggregated_column_vector), dim = 1)

    return k_ah_al_tensor, k_al_al_tensor

### SUBSET CASE ###

def aggregate_columns_rows_subsetcase(base_covariance, u):
    # hr (target) HW is the squareroot of the last dim of the base covariance
    hr_hw = int(np.sqrt(base_covariance.shape[-1])) # e.g. 60
    lr_hw = int(hr_hw / u) # e.g. 30
    lr_indices_list = range(0, lr_hw**2) # covariance has square indices (pairwise)

    # includes 0 and last index and intevals are u * hr_hw wide
    lr_row_breaks = np.linspace(0, hr_hw**2, num = int(lr_hw + 1), dtype = int)

    # Initialise empty dictionary
    lrDict = dict()
    # start with -1 as it will be updated in first it.
    lr_row_counter = -1

    for i in lr_indices_list:

        # column counter per row
        j = i%lr_hw

        # check for row breaks
        if j == 0:
            lr_row_counter += 1
            row_base = lr_row_breaks[lr_row_counter]

        lr_ranges_list = []
        for w in range(0, u):
            # each list is spanning upscaling_factor rows with upscaling_factor elements each
            lr_ranges_list.extend(range((j * u + (w * hr_hw) + row_base), (j * u + (w * hr_hw) + u + row_base)))

        lrDict[i] = lr_ranges_list

    agg_tensor_k_ah_al = torch.empty(size = (hr_hw**2, lr_hw**2))
    agg_tensor_k_al_al = torch.empty(size = (lr_hw**2, lr_hw**2))

    # go column by column
    for col_ind in lr_indices_list:
        agg_tensor_k_ah_al[:, col_ind] = torch.mean((base_covariance[:, lrDict[col_ind]]), dim = 1)
    
    # Repeat on transposed output agg_tensor_k_ah_al
    for col_ind in lr_indices_list:
        agg_tensor_k_al_al[:, col_ind] = torch.mean((torch.transpose(agg_tensor_k_ah_al, dim0 = 1, dim1 = 0)[:, lrDict[col_ind]]), dim = 1)
        
    return agg_tensor_k_ah_al, agg_tensor_k_al_al 

#######################
### PREDICTIVE MEAN ###
#######################

def predictive_mean(lr_variable_channel, k_ah_al, k_al_al, noise = 0.05, mu = 0.5):
    """ Take in argumnets to compute predictive mean and output inferred mean channel

    Args:
        lr_variable_channel (_type_): _description_
        k_ah_al (_type_): _description_
        k_al_al (_type_): _description_
        noise (float, optional): _description_. Defaults to 0.05.
        mu (float, optional): _description_. Defaults to 0.5.
    """
    # Mean channel: zero(ish) mean target_lowres grid e.g. torch.Size([1, 9, 9])
    pl_al_minus_mu = lr_variable_channel - torch.ones(size = lr_variable_channel.shape) * mu  

    # Weights
    W = torch.matmul(k_ah_al, torch.linalg.inv(k_al_al + (torch.eye(n = k_al_al.shape[-1]) * noise))) # torch.Size([2025, 81])

    # Get shape from W
    # Error with data type
    hr_inference_adjusted = torch.div(torch.matmul(W.double(), pl_al_minus_mu.reshape(-1).unsqueeze(1).double()), torch.matmul(W.double(), torch.ones(size = (W.shape[-1], 1)).double())) + mu

    # Cast into 2D shape
    hr_inference_2D = hr_inference_adjusted.reshape(int(np.sqrt(hr_inference_adjusted.shape[0])), -1)
    # t.view((int(math.sqrt(t.shape[0])), int(math.sqrt(t.shape[0]))))

    return(hr_inference_2D)

def predictive_variance(base_covariance, k_ah_al, k_al_al, noise = 0.05):
    # Assure they are tensors
    base_covariance = torch.tensor(base_covariance)
    k_ah_al = torch.tensor(k_ah_al)
    k_al_al = torch.tensor(k_al_al)

    k_inv = torch.linalg.inv(k_al_al + (torch.eye(n = k_al_al.shape[-1]) * noise))
    sigma = base_covariance - torch.matmul(k_ah_al, torch.matmul(k_inv, torch.transpose(k_ah_al, dim0 = 1, dim1 = 0)))
    return sigma


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

###############################
### CHOLESKY to speed it up ###
###############################
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

def predictive_distribution_cholesky_correction(lr_variable_channel, k_ah_ah, k_ah_al, k_al_al, noise = 0.05, mu = 0.5, batch_size = 1):
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
