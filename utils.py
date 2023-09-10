### import statements ###
import torch
import numpy as np

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

    # define upscaling function with torch https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html, default settings
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
    # Alternative: choose dim to be least common multiple (LCM): 60 (6 up), 420 (7 up), 2520
    if (((tensor.shape[-1] % upscaling_factor) != 0) or ((tensor.shape[-2] % upscaling_factor) != 0)):
        print("ACHTUNG: Upscaling is not closed/has remainder: Mean aggregation is over fields of different sizes. Consider using a different magnification factor.")
        # exit because box upscaling does not work

    # Initialise empty target tensor to append to
    target_tensor = torch.empty(size = (0, int(tensor.shape[-2] / upscaling_factor) , int(tensor.shape[-1] / upscaling_factor)))

    # define upscaling function with torch https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
    # We upscale the same amount in x & y direction: square window
    upscale = torch.nn.AvgPool2d(kernel_size = upscaling_factor, padding = 0)
    # ceil_mode = False: 45 pixel with upscale_factor 2  will be 22 in ceil_mode = False, 23 if True
    # padding would not make sense for this application: we want to reduce the size in upscaling

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


#####
def box_tensor_2_mid_points(lr_box_channels):
    lr_dim = lr_box_channels.shape[-1]

    torch.empty(size = (0, lr_dim, lr_dim))
    # y spacing (y_min/ y_max spacing)
    y_spacing = torch.abs(lr_box_channels[0, 0, 0] - lr_box_channels[0, 1, 0])

    # x spacing (x_min/ x_max spacing) (negative if other way around: x index 1 column minus x index 0 column)
    x_spacing = torch.abs(lr_box_channels[2, 0, 1] - lr_box_channels[2, 0, 0])

    # Add half y_spacing to y_min
    y_midpoints = lr_box_channels[0, :, :].unsqueeze(0) + y_spacing/2
    # Add halb x_spacing
    x_midpoints = lr_box_channels[2, :, :].unsqueeze(0) + x_spacing/2

    mid_points = torch.cat((y_midpoints, x_midpoints), dim = 0)

    return(mid_points)
