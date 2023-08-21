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

    # apply upscalin g
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