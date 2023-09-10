import torch

###############
### METRICS ###
###############

# Alternative https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
# L2 loss
def mse(image1, image2):
    # Take in two tensors of the same size and check t

    # Check
    if (image1.shape != image2.shape):
        print("Input images are not the same size. MSE can't be calculated")

    # Difference squared
    squared_error_tensor = torch.pow((image1.detach() - image2.detach()), exponent = 2)
    mean_squared_error = torch.mean(squared_error_tensor)

    return(mean_squared_error)

# same scale as input
def rmse(image1, image2):
    # Take in two tensors of the same size and check t

    # Check
    if (image1.shape != image2.shape):
        print("Input images are not the same size. MSE can't be calculated")

    # Difference squared
    squared_error_tensor = torch.pow((image1.detach() - image2.detach()), exponent = 2)
    mean_squared_error = torch.mean(squared_error_tensor)

    return(torch.sqrt(mean_squared_error))

### Image metrics ###
# https://torchmetrics.readthedocs.io/en/latest/image/peak_signal_noise_ratio.html
from torchmetrics.image import PeakSignalNoiseRatio
psnr = PeakSignalNoiseRatio()