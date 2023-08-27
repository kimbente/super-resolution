### import statements ###
import torch

### ###
def upscale_tensor(tensor, upscaling_factor):
    """ Upscaling is the opposite of downscaling: We are increasing the scale of each grid cell represented by the value by mean aggregation. 
        From in input higher resolution tensor a lower resolution tensor is returned. 

    Args:
        tensor (torch.tensor): high-res. input 2D tensor either with or without explicit first dimension.
        upscaling_factor (int): number of vertical and horizontal field to convolve over.

    Returns:
        torch.tensor: low-res. output 2D tensor either without explicit first dimension.
    """
# upscaling is the opposite of downscaling: We are increasing the scale of each grid cell represented by the value by mean aggregation.

    # check if tensor has explicit first dimension required for torch
    if (tensor.shape[0] != 1):
        # create explicit first dimension 
        tensor = tensor.unsqueeze(0)
    
    # check if input is now [1, dim1, dim2]
    if (len(list(tensor.shape)) < 3):
        print("This function can only be applied to 2D tensors.")

    if (((tensor.shape[-1] % upscaling_factor) != 0) or ((tensor.shape[-2] % upscaling_factor) != 0)):
        print("ACHTUNG: Upscaling is not closed/has remainder: Mean aggregation is over fields of different sizes. Consider using a different magnification factor.")

    # define upscaling function with torch https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
    upscale = torch.nn.AvgPool2d(kernel_size = upscaling_factor)

    # apply upscaling
    upscaled_tensor = upscale(tensor)

    # remove explicit first dimension again
    return upscaled_tensor.squeeze()


### ###
def minmax_normalise_tensor(tensor):
    """ Min-max normalises a single torch.tensor so that values lie in range [0, 1]

    Args:
        tensor (torch tensor): inputs tensor

    Returns:
        torch tensor: torch tensor with values in range [0, 1]
    """
    return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))


###############
### METRICS ###
###############

def mse(image1, image2):
    # Take in two tensors of the same size and check t

    # Check
    if (image1.shape != image2.shape):
        print("Input images are not the same size. MSE can't be calculated")

    squared_error_tensor = torch.pow(torch.div(image1.detach(), image2.detach()), exponent = 2)
    mean_squared_error = torch.mean(squared_error_tensor)

    return(mean_squared_error)

#############
### SHAPE ###
#############

def midpoint_to_box(yx_tensor):
    """_summary_

    Args:
        yx_tensor (torch.tensor): [C, H, W] tensor where C_1 is y and C_2 is x
    
    Returns:
        torch.tensor: 4-channel tensor (box-channel) y_min, y_max, x_min, x_max
    """
    # Separate both tensors
    y_tensor = yx_tensor[0, :, :] # [H, W]
    x_tensor = yx_tensor[1, :, :] # [H, W]

    # integer; y direction is reversed to obtain positive step value.
    y_step = int(y_tensor[0, 0] - y_tensor[1, 0])
    x_step = int(x_tensor[0, 1] - x_tensor[0, 0])

    ### Y MIN MAX ###
    # y_min is the lower edge of the cell.
    y_min = y_tensor - (y_step / 2)
    # y_max is the top edge of the cell.
    y_max = y_tensor + (y_step / 2)

    ### X MIN MAX ###
    # x_min is the left edge of the cell.
    x_min = x_tensor - (x_step / 2)
    # x_max is the right edge of the cell.
    x_max = x_tensor + (x_step / 2)

    return(torch.cat((y_min.unsqueeze(0),
                     y_max.unsqueeze(0),
                     x_min.unsqueeze(0), 
                     x_max.unsqueeze(0)),
                     dim = 0))


def upscale_box_tensor(tensor, upscaling_factor):
    """ Upscaling is the opposite of downscaling: We are increasing the scale of each grid cell represented by the value by mean aggregation. 
        From in input higher resolution tensor a lower resolution tensor is returned. 

    Args:
        tensor (torch.tensor): high-res. input 3D tensor where [C, H, W] where C is pixel_dim + box_dim (4)
        upscaling_factor (int): number of vertical and horizontal field to convolve over.

    Returns:
        torch.tensor: low-res. output 3D tensor with updated box variables as the last 4 channels.
    """
    pixel_dim = tensor.shape[0] - 4 
    # box_dim = 4 is implicitly "hardcoded" into the structure of this function for the rectilinear case: box edges

    # Warning if upscaling is not perfect
    if (((tensor.shape[-1] % upscaling_factor) != 0) or ((tensor.shape[-2] % upscaling_factor) != 0)):
        print("ACHTUNG: Upscaling is not closed/has remainder: Mean aggregation is over fields of different sizes. Consider using a different magnification factor.")
        # exit because box upscaling does not work
        exit

    # Initialise empty target tensor to append to
    target_tensor = torch.empty(size = (0, int(tensor.shape[-2] / upscaling_factor) , int(tensor.shape[-1] / upscaling_factor)))

    # define upscaling function with torch https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
    # We upscale the same amount in x & y direction: square window
    upscale = torch.nn.AvgPool2d(kernel_size = upscaling_factor)

    # Loop through pixel_dim and upscale each
    for i in range(0, pixel_dim):
        # need to reassign to variable name. torch.cat() is not inplace
        target_tensor = torch.cat((target_tensor, upscale(tensor[i, :, :].unsqueeze(0))), dim = 0)

    # define max_pool function with same kernel_size as upscaling
    max_pool = torch.nn.MaxPool2d(kernel_size = upscaling_factor)

    # y_min: No min pooling function: double negative
    target_tensor = torch.cat((target_tensor, (- max_pool(- tensor[pixel_dim, :, :].unsqueeze(0)))), dim = 0)
    # y_max
    target_tensor = torch.cat((target_tensor, (max_pool(tensor[(pixel_dim + 1), :, :].unsqueeze(0)))), dim = 0)
    # x_min
    target_tensor = torch.cat((target_tensor, (- max_pool(- tensor[(pixel_dim + 2), :, :].unsqueeze(0)))), dim = 0)
    # x_max
    target_tensor = torch.cat((target_tensor, (max_pool(tensor[(pixel_dim + 3), :, :].unsqueeze(0)))), dim = 0)

    return target_tensor

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