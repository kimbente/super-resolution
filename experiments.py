import pandas as pd
import numpy as np
import torch
import plotly.grid_objs as go

from utils import minmax_normalise_tensor, upscale_tensor
from covariance import aggregate_columns_rows_subsetcase
from metrics import rmse
from predictive import predictive_distribution_cholesky_correction

def run_experiment(list_of_covariance_functions, scene_bed_tensor, n_scenes):

    # n_scenes (int): number of scenes in dataset to iterate over. 400 is the maximum here
     
    n_covar_functions = len(list_of_covariance_functions)
    # Define upscaling factor. We use 60 pixel images so that upscaling is not compromised for 2 to 6.
    up_factor_list = [2, 3, 4, 5, 6]
    # Generate list of column names
    column_names = [("up_") + str(x) for x in up_factor_list] * n_covar_functions

    # Empty dataframes for losses: rows represent scenes (to guide further investigation) and columns represent upscaling factors
    proposed_rmse_df = pd.DataFrame(index = range(0, n_scenes), columns = column_names)
    baseline_rmse_df = pd.DataFrame(index = range(0, n_scenes), columns = column_names)
    # NLL only makes sense for probabilistic methods
    proposed_nll_df = pd.DataFrame(index = range(0, n_scenes), columns = column_names)

    # Log marginal likelihood placeholder
    proposed_lml_df = pd.DataFrame(index = range(0, n_scenes), columns = column_names)


    # for each upscaling_factor
    for u_index, u in enumerate(up_factor_list):
        # for each scene (convert to batches later)
        for i in range(0, n_scenes):
            # Normalise target bed topography scene and create explicit first dim, e.g. torch.Size([1, 60, 60])
            target_ground_truth = minmax_normalise_tensor(scene_bed_tensor[i, 0, :, :]).unsqueeze(0)
            # Extract target height-width: last dimension (since H == W)
            target_hw = target_ground_truth.shape[-1]
            lr_hw = int(target_hw / u)
                
            # Upscale (Increase scale of each pixel, reduce resolution) to generate low-res. input
            lr_bed = upscale_tensor(target_ground_truth, upscaling_factor = u)

            ### BASELINE ###
            # For torch grid resample function: Normalised grid as image input [N, C, H, W] where N = 1 and C = 1. 
            # H_in and W_in are implicit: corners of midpoints are assumed to me -1, -1 (top left) and 1, 1 (bottom right).
            # Always first dim
            lr_input_grid = lr_bed.unsqueeze(0).unsqueeze(0)

            # Assuming boundries are the same for both: outer boundries are [-1, 1] for both hr and lr
            d = torch.tensor(np.linspace(start = (-1.0 + (2/target_hw)/2) , stop = (1.0 - (2/target_hw)/2), num = target_hw))
            meshx, meshy = torch.meshgrid((d, d), indexing = "xy")
            # x,y order
            target_grid = torch.stack((meshx, meshy), 2)
            target_grid = target_grid.unsqueeze(0) # add batch dim
                
            # Border works much better than zero: Since we sample at a higher resolution than the input we sample e.g. left of the leftmost HR location
            hr_bilinear = torch.nn.functional.grid_sample(lr_input_grid.float(), target_grid.float(), mode = 'bilinear', padding_mode = 'border', align_corners = False)
            hr_bilinear = hr_bilinear.squeeze()

            ### BASELINE LOSS ###
            baseline_rmse_df.iloc[i, u_index] = rmse(hr_bilinear, target_ground_truth.squeeze()).numpy().item()

            ### PROPOSED ###
            # Normalisation of high-resolution auxiliary channel to compute the base_covariance
            hr_aux = minmax_normalise_tensor(scene_bed_tensor[i, 1, :, :]).unsqueeze(0)

            # Iterate through covar functions
            for c_index, c in enumerate(list_of_covariance_functions):

                ### Base covariance ###
                base_covariance = c(hr_aux)
                k_ah_al_tensor, k_al_al_tensor = aggregate_columns_rows_subsetcase(base_covariance = base_covariance, u = u)

                ### MEAN RECONSTRUCTION ###
                hr_mean_inferred, hr_covariance_inferred, lml = predictive_distribution_cholesky_correction(lr_bed.unsqueeze(0), base_covariance, k_ah_al_tensor, k_al_al_tensor, 
                                                                                noise = torch.tensor(0.05), mu = torch.tensor(0.5))

                # Save log marginal likelihood
                proposed_lml_df.iloc[i, (u_index * (c_index + 1))] = lml

                ### LOSS ###
                # inplace mutation of row i and column u (upscale_factor) of df
                proposed_rmse_df.iloc[i, (u_index * (c_index + 1))] = rmse(hr_mean_inferred, target_ground_truth).numpy().item()

                # NLL; store items not tensors in df
                # extract variances (on diagonal) from covariance matrix and reshape to square
                hr_variance_inferred = torch.diagonal(hr_covariance_inferred).reshape(1, target_hw, -1)
                proposed_nll_df.iloc[i, (u_index * (c_index + 1))] = torch.nn.functional.gaussian_nll_loss(hr_mean_inferred, target_ground_truth, hr_variance_inferred, full = False, eps = 1e-06, reduction = 'mean').numpy().item()
    
    return(proposed_rmse_df, baseline_rmse_df, proposed_nll_df, proposed_lml_df)




def visualise_results(proposed_rmse_df, baseline_rmse_df, proposed_nll_df, proposed_lml_df, n_scenes, domain_name):

    # RMSE
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(range(2, 6 + 1)), y =  baseline_rmse_df.mean(), mode = 'lines+markers', name = "Bilinear baseline"))
    fig.add_trace(go.Scatter(x = list(range(2, 6 + 1)), y = proposed_rmse_df.iloc[:, 0:5].mean(), mode = 'lines+markers', name = "Proposed algorithm"))
    fig.update_layout(title = 'Reconstruction loss [RMSE] of proposed vs. baseline - {} scenes near domain {}'.format(n_scenes, domain_name))
    fig.update_xaxes(title_text = 'Upscaling factor')
    fig.update_yaxes(title_text = 'RMSE')
    fig.show()

    # NLL
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(range(2, 6 + 1)), y = proposed_nll_df.iloc[:, 0:5].mean(), mode = 'lines+markers', name = "Proposed algorithm"))
    fig.update_layout(title = "Reconstruction loss [NLL]")
    fig.update_xaxes(title_text = 'Upscaling factor')
    fig.update_yaxes(title_text = 'NLL')
    fig.show()

    # LML
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(range(2, 6 + 1)), y = proposed_lml_df.iloc[:, 0:5].mean(), mode = 'lines+markers', name = "Proposed algorithm"))
    fig.update_layout(title = "Reconstruction Log marginal likelihood [LML] (larger is better)")
    fig.update_xaxes(title_text = 'Upscaling factor')
    fig.update_yaxes(title_text = 'LML')
    fig.show()