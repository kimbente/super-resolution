class SuperresolutionGP_lowerRAM:
    # Initialise with training data tensor/test tensor
    def __init__(self, train_tensor):
        self.n_train = train_tensor.shape[0] # Input comes in [N, C, H, W] format
        self.hr_hw = train_tensor.shape[-1] # Last dimension is W (width) which is equal to H
        self.up_factors = np.array([2, 3, 4, 5, 6]) # hardcoded for now. These 
        self.n_up_factors = len(self.up_factors)
        self.up_lr_hw = (np.repeat(self.hr_hw, self.n_up_factors) / self.up_factors).astype(int) # LR input dimensionalities

        # CHECKPOINT: Test if self.hr_hw can be divided by ALL upscaling factors without a remainder. This implementation only works for easy subset cases.
        if (np.all((self.hr_hw % self.up_factors) == np.zeros(shape = len(self.up_factors))) != True):
            raise ValueError("Upscaling factors and Height/Width of pixel are not compatible. Use a different algorithm or a subset of compatible upscaling factors.") 

        # [N, C, H_h, W_h]
        self.train_bed_ground_truth = train_tensor[:, 0, :, :].unsqueeze(1).to(device) # channel 0 contains bed, retain channel dim
        # [N, C, H_h, W_h]
        self.train_sur_hr_aux = train_tensor[:, 1, :, :].unsqueeze(1).to(device) # channel 1 contains surface, retain channel dim      
    
        # Upscale and retain list of tensors (varying dimensionality)
        # [Up *] with tensors [N, C, H_l, W_l]
        self.train_bed_lr = self.upscale(self.train_bed_ground_truth) #

        # Hyperparameters initialisation
        self.gp_hyper_lambda_s = 0.4
        self.gp_hyper_lambda_p = 0.7
        self.gp_hyper_sigma_f = 1.0 # amplitude

        # Noise and constant mean
        self.gp_hyper_noise = 0.05
        self.gp_mu = 0.5

        # K_ah_ah, K_ah_al, K_al_al
        # self.train_hr_aux_basecovariance = None
        # self.per_upfactor_n_agg_tensor_k_ah_al = None
        # self.per_upfactor_n_agg_tensor_k_al_al = None

        # Create variables if we want to investigate these
        # self.spatial_base_covariance_matrix = None
        # self.pixel_base_covariance_matrix = None

    #############
    ### UTILS ###
    #############

    def upscale(self, bed_ground_truth):
        # Upscale (Increase scale of each pixel, reduce resolution) to articially generate low-res. input. controlled experiment.
        # Empty list of tensors (varying dims - can't stack tensors)
        train_lr = []
        for u in self.up_factors:
            # define upscaling function with torch https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html, default settings
            upscaling_function = torch.nn.AvgPool2d(kernel_size = u)
            # appending is in-place: no reassignment needed
            train_lr.append(upscaling_function(bed_ground_truth))
        return train_lr
    
    def set_hypers(self, hyperparameters):
        # Assign new params
        self.gp_hyper_lambda_s = hyperparameters[0]
        self.gp_hyper_lambda_p = hyperparameters[1]
        self.gp_hyper_sigma_f = hyperparameters[2]

    def predict_and_evaluate_training(self):
        # Can be called after updating the hyperparameters
        # Base covariance: Generate base covariance using currently stored hypers, runs on self.train_sur_hr_aux
        """Better for memory if we take in on this level"""

        base_covariance_matrix = self.composite_base_covariance(self.train_sur_hr_aux)

        # Aggregate: Updates stored covariance tensors, using self.train_hr_aux_basecovariance from above
        k_ah_al, k_al_al = self.aggregate_base_covariance(base_covariance_matrix)

        # Predictand return lml. Uses the above tensors: k_ah_ah, k_ah_al, k_al_al so updates were needed
        mean_UNCHW, covariance_UNCHHWW, lml_UNC = self.predictive_distribution(base_covariance_matrix, k_ah_al, k_al_al, self.train_bed_lr)

        return mean_UNCHW, covariance_UNCHHWW, lml_UNC

    def predict_and_rmse(self):
        # Can be called after updating the hyperparameters
        # Base covariance: Generate base covariance using currently stored hypers, runs on self.train_sur_hr_aux
        """Better for memory if we take in on this level"""
        
        base_covariance_matrix = self.composite_base_covariance(self.train_sur_hr_aux)

        # Aggregate: Updates stored covariance tensors, using self.train_hr_aux_basecovariance from above
        k_ah_al, k_al_al = self.aggregate_base_covariance(base_covariance_matrix)

        # Predictand return lml. Uses the above tensors: k_ah_ah, k_ah_al, k_al_al so updates were needed
        mean_UNHW, _, _ = self.predictive_distribution(base_covariance_matrix, k_ah_al, k_al_al, self.train_bed_lr)
        # Create channel at second position
        mean_UNCHW = mean_UNHW.unsqueeze(2) # NCHW (missing first Up channel)
        return self.rmse(self.train_bed_ground_truth, mean_UNCHW)
    
    #################
    ### BASELINE ####
    #################

    def bilinear_interpolation_baseline(self, bed_lr):
        """Baseline method

        Args:
            bed_lr (list of torch.tensors): list of torch tensors of different dimensionalities
        """
        # bed_lr is a list of tensors

        # Create one target grid: outer boundries of each scene are the same for both hr and lr: [-1, 1]
        d = torch.tensor(np.linspace(start = (-1.0 + (2/self.hr_hw)/2) , stop = (1.0 - (2/self.hr_hw)/2), num = self.hr_hw))
        meshx, meshy = torch.meshgrid((d, d), indexing = "xy") # create mesh
        target_grid = torch.stack((meshx, meshy), 2) # x,y order
        target_grid = target_grid.unsqueeze(0) # add batch dim: torch.Size([1, self.hr_hw, self.hr_hw, 2])

        # Create empty placeholder tensor of shape [Up = 0, N, C = 1, H, W]
        up_n_bed_hr = torch.empty(size = (0, bed_lr[0].shape[0], 1, self.hr_hw, self.hr_hw))

        # Simple baseline in bilinear interpolation using torch https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html 
        for u_index, u in enumerate(self.up_factors):

            # copy target grid n times
            n = bed_lr[u_index].shape[0]
            n_target_grids = torch.tile(target_grid, dims = (n, 1, 1, 1))

            hr_bilinear = torch.nn.functional.grid_sample(bed_lr[u_index].float(), n_target_grids.float(), mode = 'bilinear', padding_mode = 'border', align_corners = False)
            up_n_bed_hr = torch.cat((up_n_bed_hr, hr_bilinear.unsqueeze(0)), dim = 0) # generate explicit first dim for all up_factors

        return(up_n_bed_hr) # [Up, N, C = mean, H, W]
    
    ###############
    ### METRICS ###
    ###############

    def rmse(self, ground_truth, predictions):
        """_summary_

        Args:
            ground_truth (_type_): [N, 1, H, W] - will be copied for all up_factors
            predictions (_type_): [Up, N, 1, H, W]

        Returns:
            [Up, N, 1]
        """
        n = predictions.shape[0]
        # Copy for n_Up_factors into [Up, N, 1, H, W]
        n_ground_truth = torch.tile(ground_truth.unsqueeze(0), dims = (n, 1, 1, 1, 1))

        error = torch.sub(n_ground_truth, predictions) # subtract elementwise
        squared_error = torch.pow(error, exponent = 2) # square error to eliminate negatives
        mean_squared_error = torch.mean(squared_error, dim = (-2, -1)) # mean across H and W (last two dim)
        root_mean_squared_error = torch.sqrt(mean_squared_error)
        
        # ToDo: mean over N
        return root_mean_squared_error # [Up, N, C = RMSE]
    
    def nll(self):
        # TBC
        return 0
    
    # LML is implemented in the prediction loop
    
    ##################
    ### COVARIANCE ###
    ##################

    def composite_base_covariance(self, aux_tensor):
        # spatial_base_covariance does not depend on any inputs

        # Initialise placeholder for all batches
        n_scaled_base_covariance_matrix = torch.empty(size = (0, aux_tensor.shape[-1] * aux_tensor.shape[-1], aux_tensor.shape[-1] * aux_tensor.shape[-1]))

        composite_dataloader = torch.utils.data.DataLoader(aux_tensor, batch_size = 4, shuffle = False)

        for batch in composite_dataloader:
            spatial_base_covariance_matrix = self.spatial_base_covariance(batch.shape[0])
            pixel_base_covariance_matrix = self.pixel_base_covariance(batch)
            
            # Element-wise multiplication, Hadamard product, AND operation
            product_base_covar = torch.mul(spatial_base_covariance_matrix, pixel_base_covariance_matrix)
            
            # multiply all elements in matrix with same scalar
            scaled_base_covariance_matrix = torch.mul(product_base_covar, self.gp_hyper_sigma_f)
            
            # Concat with batches
            n_scaled_base_covariance_matrix = torch.cat((n_scaled_base_covariance_matrix, scaled_base_covariance_matrix), dim = 0)

        return n_scaled_base_covariance_matrix
    
    def only_spatial_base_covariance(self):
        # Fast do no batching needed
        spatial_base_covariance_matrix = self.spatial_base_covariance(self.n_train)

        return spatial_base_covariance_matrix

    def spatial_base_covariance(self, n_batch):
        # smooth, sparse, local
        # lambda_s is the hp that controls the receptive field
        # Same for all n so we calculate it once and create copies

        n = n_batch

        # Normalised mid_points of all picels in scene
        xs = torch.arange(0, (self.hr_hw)).repeat(self.hr_hw, 1)
        ys = xs.mT
        # x and y dim of midpoints from 0 to 1 
        mid_points_norm = torch.cat((ys.unsqueeze(0), xs.unsqueeze(0)), dim = 0)/(self.hr_hw -1) # torch.Size([2, 60, 60])

        mid_points_flat = torch.flatten(mid_points_norm, start_dim = -2) # torch.Size([2, 3600])
        # broadcast to calculate pairwise distance
        dist = torch.sub(mid_points_flat.unsqueeze(-1), mid_points_flat.unsqueeze(-2))
        # square all distances
        dist_square = torch.pow(dist, exponent = 2)
        # sum x direction dist and y direction dist (Euclidean dist)
        dist_sum = torch.sum(dist_square, dim = 0)
        dist_euc = torch.sqrt(dist_sum) # torch.Size([3600, 3600]), Pythagoras, max is 1.4142 (corners, sqrt(2))

        Z = torch.div(dist_euc, self.gp_hyper_lambda_s) # divide by scalar, lambda_s is threshold for 0 covariance
        # Mask large Z's (too-far-away values) with nan before replacing values with zero later
        Z[Z >= 1] = float('nan')
        # first term pushes small distances to 0 and distances near 1 close to zero
        # second terms is clipped at 1 so that small distances will approach 1
        prod_term1 = torch.pow(torch.add(-Z, 1), exponent = 3) # (1 - Z) = (-Z + 1)
        prod_term2 = torch.add(torch.mul(Z, 3), 1.)
        cov_matrix = torch.mul(prod_term1, prod_term2) # elementwise multiplication
        # fill nan's with zero
        cov_matrix[torch.isnan(cov_matrix)] = 0.0
        cov_matrix = cov_matrix.unsqueeze(0) # torch.Size([0, 3600, 3600])

        # Create n copies
        n_cov_matrix = torch.tile(cov_matrix, dims = (n, 1, 1)) # torch.Size([n, 3600, 3600])

        # torch.Size([300, 3600, 3600])
        return n_cov_matrix

    def pixel_base_covariance(self, aux_tensor):
        # coupling(decoupling) of similar(dissimilar) pixel values, non-stationary
        # lambda_p is the lengthscale of the RBF kernel, default is 0.6931 in GPytorch
        # for 300 scenes this has a wall time of 8:20 min on MacPro

        # Reduce Channel dim
        hr_aux = torch.flatten(aux_tensor, start_dim = 1, end_dim = 2)

        n_cov_matrix = torch.empty(size = (0, hr_aux.shape[-1] * hr_aux.shape[-1], hr_aux.shape[-1] * hr_aux.shape[-1]))

        # Apply batchwise so it does not crash
        """Reduce inner batch to reduce comp. load"""
        dataloader = torch.utils.data.DataLoader(hr_aux, batch_size = 4, shuffle = False)
        for batch_hr_aux in dataloader:
            # flatten x and y
            batch_hr_aux_flat = torch.flatten(batch_hr_aux, start_dim = -2)
            # broadcast for pairwise dist (in pixel dim)
            dist = torch.sub(batch_hr_aux_flat.unsqueeze(-1), batch_hr_aux_flat.unsqueeze(-2))
            # direction of distance does not matter
            dist_squared = torch.pow(dist, exponent = 2)
            dist_squared_scaled = torch.div(dist_squared, (2 * torch.pow(torch.tensor(self.gp_hyper_lambda_p), exponent = 2)))

            # 1 at zero distance
            rbf = torch.exp( - dist_squared_scaled)

            n_cov_matrix = torch.cat((n_cov_matrix, rbf), dim = 0)
        
        return n_cov_matrix
    
    def aggregate_base_covariance(self, base_covariance_matrix):

        # Go through one at a time
        n_base_covars = base_covariance_matrix.shape[0]

        per_upfactor_n_agg_tensor_k_ah_al = []
        per_upfactor_n_agg_tensor_k_al_al = []

        for u_index, u in enumerate(self.up_factors):
    
            # retrieve lr_hw from self
            lr_hw = self.up_lr_hw[u_index]
            lr_indices_list = range(0, lr_hw**2) # covariance has square indices (pairwise)

            # includes 0 and last index and intevals are u * hr_hw wide
            lr_row_breaks = np.linspace(0, self.hr_hw**2, num = int(lr_hw + 1), dtype = int)

            # Placeholders
            n_agg_tensor_k_ah_al = torch.empty(size = (0, self.hr_hw**2, lr_hw**2))
            n_agg_tensor_k_al_al = torch.empty(size = (0, lr_hw**2, lr_hw**2))

            for n in range(0, n_base_covars):
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
                        lr_ranges_list.extend(range((j * u + (w * self.hr_hw) + row_base), (j * u + (w * self.hr_hw) + u + row_base)))

                    lrDict[i] = lr_ranges_list

                agg_tensor_k_ah_al = torch.empty(size = (self.hr_hw**2, lr_hw**2))
                agg_tensor_k_al_al = torch.empty(size = (lr_hw**2, lr_hw**2))

                # go column by column
                for col_ind in lr_indices_list:
                    # Only select n 
                    """ERROR?"""
                    agg_tensor_k_ah_al[:, col_ind] = torch.mean((base_covariance_matrix[n, :, lrDict[col_ind]]), dim = 1)
    
                # Repeat on transposed output agg_tensor_k_ah_al
                for col_ind in lr_indices_list:
                    agg_tensor_k_al_al[:, col_ind] = torch.mean((torch.transpose(agg_tensor_k_ah_al, dim0 = 1, dim1 = 0)[:, lrDict[col_ind]]), dim = 1)

                # Concatinate
                n_agg_tensor_k_ah_al = torch.cat((n_agg_tensor_k_ah_al, agg_tensor_k_ah_al.unsqueeze(0)), dim = 0)
                n_agg_tensor_k_al_al = torch.cat((n_agg_tensor_k_al_al, agg_tensor_k_al_al.unsqueeze(0)), dim = 0)
            
            # Append to list of 
            per_upfactor_n_agg_tensor_k_ah_al.append(n_agg_tensor_k_ah_al)
            per_upfactor_n_agg_tensor_k_al_al.append(n_agg_tensor_k_al_al)

        return per_upfactor_n_agg_tensor_k_ah_al, per_upfactor_n_agg_tensor_k_al_al
    
    def predictive_distribution(self, k_ah_ah, k_ah_al, k_al_al, train_lr):
        """Calculation of the mean and variance of the Gaussian predictive distribution as well as of Log Marginal Likelihood. 
    Here we implement the inversion through the Cholesky decomposition [torch.linalg.cholesky()] followed by the inversion [torch.cholesky_inverse(L)]. 
    This is slighly different to the Algorithm 2.10 proposed in Rasmussen & Williams but more suitable to implement the weight correction performed by Reid et al.
    
    k_ah_ah is a torch tensor.
    k_ah_al is a list of tensors.
    k_al_al is a list of tensors.
    train_lr is a 
    """
        
        # Go through one at a time
        n_base_covars = k_ah_ah.shape[0]

        # empty tensor
        upfactors_n_mean = torch.empty(size = (0, n_base_covars, self.hr_hw, self.hr_hw ))
        upfactors_n_covariance = torch.empty(size = (0, n_base_covars, self.hr_hw**2, self.hr_hw**2))
        upfactors_n_lml = torch.empty(size = (0, n_base_covars))

        for u_index, u in enumerate(self.up_factors):
            n_mean = torch.empty(size = (0, self.hr_hw, self.hr_hw)) # mean field
            n_covariance = torch.empty(size = (0, self.hr_hw**2, self.hr_hw**2))
            n_lml = torch.empty(size = (0, ))

            for n in range(0, n_base_covars):
                # Calculate inverse via the Cholesky decomposition (Lower Triangular) of the lower input resolution
                L = torch.linalg.cholesky(k_al_al[u_index][n] + (torch.eye(n = k_al_al[u_index][n].shape[-1]) * self.gp_hyper_noise))
                k_inv = torch.cholesky_inverse(L)

                #### Mean ###
                pl_al_minus_mu = train_lr[u_index][n] - torch.mul(torch.ones(size = train_lr[u_index][n].shape), self.gp_mu)
                W = torch.matmul(k_ah_al[u_index][n], k_inv) 

                # Correction, torch.div is element-wise vision
                mean_flat = torch.div(torch.matmul(W, pl_al_minus_mu.reshape(-1).unsqueeze(1)), torch.matmul(W, torch.ones(size = (W.shape[-1], 1)))) + self.gp_mu
                # cast into 2D shape
                mean = mean_flat.reshape(1, int(np.sqrt(mean_flat.shape[0])), -1)

                ### Sigma ###
                # .mT transposes the last two dims of a matrix
                sigma = k_ah_ah[n] - torch.matmul(k_ah_al[u_index][n], torch.matmul(k_inv, k_ah_al[u_index][n].mT))

                ### LML ###
                # 2.30 in Rasmussen
                # https://d2l.ai/chapter_gaussian-processes/gp-inference.html
    
                # Term1: Kernel term
                term1 = torch.mul(torch.matmul(train_lr[u_index][n].mT.reshape(1, -1), torch.matmul(k_inv, train_lr[u_index][n].reshape(-1, 1))), 0.5)
    
                # Term2: Determinant term. 2 and 0.5 cancel each other out, sum in log space, log makes values negative
                term2 = torch.sum(torch.log(torch.diagonal(L)))
                # Need trick https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
                # Does not work: term2 = torch.mul(torch.log(torch.linalg.det(k_al_al + (torch.eye(n = k_al_al.shape[-1]) * noise))), 0.5).reshape(1, 1)
                # Flatten shape and extract n, natural log
    
                # Term3: Constant term: Only extracts shape from train_lr
                term3 = (torch.log(torch.tensor(2 * math.pi)) * torch.tensor(train_lr[u_index][n].reshape(-1).shape[0] * 0.5)).reshape((1, 1)) # n
                lml = - term1 - term2 - term3

                # Append term
                n_mean = torch.cat((n_mean, mean), dim = 0)
                n_covariance = torch.cat((n_covariance, sigma.unsqueeze(0)), dim = 0)
                n_lml = torch.cat((n_lml, lml.squeeze(1)), dim = 0)
            
            upfactors_n_mean = torch.cat((upfactors_n_mean, n_mean.unsqueeze(0)), dim = 0)
            upfactors_n_covariance = torch.cat((upfactors_n_covariance, n_covariance.unsqueeze(0)), dim = 0)
            upfactors_n_lml = torch.cat((upfactors_n_lml, n_lml.unsqueeze(0)), dim = 0)

        return upfactors_n_mean, upfactors_n_covariance, upfactors_n_lml

###################
###################
### 2nd version ###
###################
###################

class SuperresolutionGP:
    # Initialise with training data tensor/test tensor
    def __init__(self, train_tensor, scaling_bool = True):
        self.n_train = train_tensor.shape[0] # Input comes in [N, C, H, W] format
        self.hr_hw = train_tensor.shape[-1] # Last dimension is W (width) which is equal to H
        self.up_factors = np.array([2,]) # hardcoded for now. These 
        self.n_up_factors = len(self.up_factors)
        self.up_lr_hw = (np.repeat(self.hr_hw, self.n_up_factors) / self.up_factors).astype(int) # LR input dimensionalities

        self.scaling = scaling_bool

        # CHECKPOINT: Test if self.hr_hw can be divided by ALL upscaling factors without a remainder. This implementation only works for easy subset cases.
        if (np.all((self.hr_hw % self.up_factors) == np.zeros(shape = len(self.up_factors))) != True):
            raise ValueError("Upscaling factors and Height/Width of pixel are not compatible. Use a different algorithm or a subset of compatible upscaling factors.") 

        # Store scaling parameters to translate errors and predictions back into the original space
        # Initialised as None, but will be overwritten by self.scale function
        self.scaling_global_range = None
        self.scaling_global_min = None

        # Input Tensors. 
        if self.scaling == True:
            self.train_bed_ground_truth = self.scale(train_tensor[:, 0, :, :].unsqueeze(1)) # channel 0 contains bed, retain channel dim
            self.train_sur_hr_aux = self.scale(train_tensor[:, 1, :, :].unsqueeze(1)) # channel 1 contains surface, retain channel dim
        # (scaling == False) if we maybe want to scale outside of the object
        else:
            self.train_bed_ground_truth = train_tensor[:, 0, :, :].unsqueeze(1) # channel 0 contains bed, retain channel dim
            self.train_sur_hr_aux = train_tensor[:, 1, :, :].unsqueeze(1) # channel 1 contains surface, retain channel dim      
    
        # Upscale and retain list of tensors (varying dimensionality)
        self.train_bed_lr = self.upscale(self.train_bed_ground_truth)

        # Hyperparameters initialisation
        self.gp_hyper_lambda_s = 1.2
        self.gp_hyper_lambda_p = 1.4
        self.gp_hyper_sigma_f = 0.2 # amplitude

        # Noise and constant mean
        self.gp_hyper_noise = 0.05
        self.gp_mu = 0.5

        # K_ah_ah, K_ah_al, K_al_al
        self.train_hr_aux_basecovariance = None
        self.per_upfactor_n_agg_tensor_k_ah_al = None
        self.per_upfactor_n_agg_tensor_k_al_al = None

        # Create variables if we want to investigate these
        self.spatial_base_covariance_matrix = None
        self.pixel_base_covariance_matrix = None

        # Baseline, fast to fun
        self.train_baseline_predictions = self.bilinear_interpolation_baseline(self.train_bed_lr) # torch.Size([5, 300, 1, 60, 60]) with [Up, N, C = mean, H, W]
        self.train_baseline_rmse = self.rmse(self.train_bed_ground_truth, self.train_baseline_predictions) # torch.Size([5, 300, 1]) with [Up, N, C = RMSE]

    #############
    ### UTILS ###
    #############

    def scale(self, input_tensor):
        """Min-max normalisation that projects inputs into [0, 1] (inclusive, inclusive) range.
        Can be turned off using self.scaling boolean.
        Currently only implemented to work on 1-Channel inputs.

        Args:
            input_tensor (torch.tensor): input tensor

        Returns:
            (torch.tensor): scaled version of input. 
        """
        # Update global parameter
        self.scaling_global_min = torch.min(input_tensor)
        global_max = torch.max(input_tensor)
        self.scaling_global_range = (global_max - self.scaling_global_min)

        minmax_scaled = (input_tensor - self.scaling_global_min) / self.scaling_global_range

        # Optional: save global values to self to apply same standardisation for test
        return minmax_scaled

    def upscale(self, bed_ground_truth):
        # Upscale (Increase scale of each pixel, reduce resolution) to articially generate low-res. input. controlled experiment.
        # Empty list of tensors (varying dims - can't stack tensors)
        train_lr = []
        for u in self.up_factors:
            # define upscaling function with torch https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html, default settings
            upscaling_function = torch.nn.AvgPool2d(kernel_size = u)
            # appending is in-place: no reassignment needed
            train_lr.append(upscaling_function(bed_ground_truth))
        return train_lr
    
    def set_hypers(self, hyperparameters):
        # Assign new params
        self.gp_hyper_lambda_s = hyperparameters[0]
        self.gp_hyper_lambda_p = hyperparameters[1]
        self.gp_hyper_sigma_f = hyperparameters[2]

    def predict_and_evaluate_training(self):
        # Can be called after updating the hyperparameters
        # Base covariance: Generate base covariance using currently stored hypers, runs on self.train_sur_hr_aux
        self.train_hr_aux_basecovariance = self.composite_base_covariance()

        # Aggregate: Updates stored covariance tensors, using self.train_hr_aux_basecovariance from above
        self.per_upfactor_n_agg_tensor_k_ah_al, self.per_upfactor_n_agg_tensor_k_al_al = self.aggregate_base_covariance()

        # Predictand return lml. Uses the above tensors: k_ah_ah, k_ah_al, k_al_al so updates were needed
        upfactors_n_mean, upfactors_n_covariance, upfactors_n_lml = self.predictive_distribution()

        return upfactors_n_mean, upfactors_n_covariance, upfactors_n_lml
    
    #################
    ### BASELINE ####
    #################

    def bilinear_interpolation_baseline(self, bed_lr):
        """Baseline method

        Args:
            bed_lr (list of torch.tensors): list of torch tensors of different dimensionalities
        """
        # bed_lr is a list of tensors

        # Create one target grid: outer boundries of each scene are the same for both hr and lr: [-1, 1]
        d = torch.tensor(np.linspace(start = (-1.0 + (2/self.hr_hw)/2) , stop = (1.0 - (2/self.hr_hw)/2), num = self.hr_hw))
        meshx, meshy = torch.meshgrid((d, d), indexing = "xy") # create mesh
        target_grid = torch.stack((meshx, meshy), 2) # x,y order
        target_grid = target_grid.unsqueeze(0) # add batch dim: torch.Size([1, self.hr_hw, self.hr_hw, 2])

        # Create empty placeholder tensor of shape [Up, N, C = 1, H, W]
        up_n_bed_hr = torch.empty(size = (0, bed_lr[0].shape[0], 1, self.hr_hw, self.hr_hw))

        # Simple baseline in bilinear interpolation using torch https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html 
        for u_index, u in enumerate(self.up_factors):
            # copy target grid n times
            n = bed_lr[u_index].shape[0]
            n_target_grids = torch.tile(target_grid, dims = (n, 1, 1, 1))

            hr_bilinear = torch.nn.functional.grid_sample(bed_lr[u_index].float(), n_target_grids.float(), mode = 'bilinear', padding_mode = 'border', align_corners = False)
            up_n_bed_hr = torch.cat((up_n_bed_hr, hr_bilinear.unsqueeze(0)), dim = 0) # generate explicit first dim for all up_factors

        return(up_n_bed_hr)
    
    ###############
    ### METRICS ###
    ###############

    def rmse(self, ground_truth, predictions):
        """_summary_

        Args:
            ground_truth (_type_): _description_
            predictions (_type_): _description_
        """
        n = predictions.shape[0]
        n_ground_truth = torch.tile(ground_truth.unsqueeze(0), dims = (n, 1, 1, 1, 1))
        error = torch.sub(n_ground_truth, predictions) # subtract elementwise
        squared_error = torch.pow(error, exponent = 2) # square error to eliminate negatives
        mean_squared_error = torch.mean(squared_error, dim = (-2, -1)) # mean across H and W (last two dim)
        root_mean_squared_error = torch.sqrt(mean_squared_error)
        return(root_mean_squared_error) # torch.Size([5, 300, 1])
    
    def nll(self):
        # TBC
        return 0
    
    # LML is implemented in the prediction loop
    
    ##################
    ### COVARIANCE ###
    ##################

    def composite_base_covariance(self):
        self.spatial_base_covariance_matrix = self.spatial_base_covariance()
        self.pixel_base_covariance_matrix = self.pixel_base_covariance()

        # element-wise multiplication
        product_base_covar = torch.mul(self.spatial_base_covariance_matrix, self.pixel_base_covariance_matrix)
        # multiply with scalar
        scaled_base_covar = torch.mul(product_base_covar, self.gp_hyper_sigma_f)

        return scaled_base_covar
    
    def only_spatial_base_covariance(self):
        # Overwrite base_covariance
        self.train_hr_aux_basecovariance = self.spatial_base_covariance(self.train_sur_hr_aux )

    def spatial_base_covariance(self):
        # smooth, sparse, local
        # lambda_s is the hp that controls the receptive field

        # input: hr_aux only for shape: extract n to create n copies
        n = self.train_sur_hr_aux .shape[0]

        # Normalised mid_points of all picels in scene
        xs = torch.arange(0, (self.hr_hw)).repeat(self.hr_hw, 1)
        ys = xs.mT
        # x and y dim of midpoints from 0 to 1 
        mid_points_norm = torch.cat((ys.unsqueeze(0), xs.unsqueeze(0)), dim = 0)/(self.hr_hw -1) # torch.Size([2, 60, 60])

        mid_points_flat = torch.flatten(mid_points_norm, start_dim = -2) # torch.Size([2, 3600])
        # broadcast to calculate pairwise distance
        dist = torch.sub(mid_points_flat.unsqueeze(-1), mid_points_flat.unsqueeze(-2))
        # square all distances
        dist_square = torch.pow(dist, exponent = 2)
        # sum x direction dist and y direction dist (Euclidean dist)
        dist_sum = torch.sum(dist_square, dim = 0)
        dist_euc = torch.sqrt(dist_sum) # torch.Size([3600, 3600]), Pythagoras, max is 1.4142 (corners, sqrt(2))

        Z = torch.div(dist_euc, self.gp_hyper_lambda_s) # divide by scalar, lambda_s is threshold for 0 covariance
        # Mask large Z's (too-far-away values) with nan before replacing values with zero later
        Z[Z >= 1] = float('nan')
        # first term pushes small distances to 0 and distances near 1 close to zero
        # second terms is clipped at 1 so that small distances will approach 1
        prod_term1 = torch.pow(torch.add(-Z, 1), exponent = 3) # (1 - Z) = (-Z + 1)
        prod_term2 = torch.add(torch.mul(Z, 3), 1.)
        cov_matrix = torch.mul(prod_term1, prod_term2) # elementwise multiplication
        # fill nan's with zero
        cov_matrix[torch.isnan(cov_matrix)] = 0.0
        cov_matrix = cov_matrix.unsqueeze(0) # torch.Size([0, 3600, 3600])

        # Create n copies
        n_cov_matrix = torch.tile(cov_matrix, dims = (n, 1, 1)) # torch.Size([n, 3600, 3600])

        # torch.Size([300, 3600, 3600])
        return n_cov_matrix

    def pixel_base_covariance(self):
        # coupling(decoupling) of similar(dissimilar) pixel values, non-stationary
        # lambda_p is the lengthscale of the RBF kernel, default is 0.6931 in GPytorch
        # for 300 scenes this has a wall time of 8:20 min on MacPro

        # Reduce Channel dim
        hr_aux = torch.flatten(self.train_sur_hr_aux, start_dim = 1, end_dim = 2)

        n_cov_matrix = torch.empty(size = (0, hr_aux.shape[-1] * hr_aux.shape[-1], hr_aux.shape[-1] * hr_aux.shape[-1]))

        # Apply batchwise so it does not crash
        dataloader = torch.utils.data.DataLoader(hr_aux, batch_size = 8, shuffle = False)
        for batch_hr_aux in dataloader:
            # flatten x and y
            batch_hr_aux_flat = torch.flatten(batch_hr_aux, start_dim = -2)
            # broadcast for pairwise dist (in pixel dim)
            dist = torch.sub(batch_hr_aux_flat.unsqueeze(-1), batch_hr_aux_flat.unsqueeze(-2))
            # direction of distance does not matter
            dist_squared = torch.pow(dist, exponent = 2)
            dist_squared_scaled = torch.div(dist_squared, (2 * torch.pow(torch.tensor(self.gp_hyper_lambda_p), exponent = 2)))

            # 1 at zero distance
            rbf = torch.exp(- dist_squared_scaled)

            n_cov_matrix = torch.cat((n_cov_matrix, rbf), dim = 0)
        
        return n_cov_matrix
    
    def aggregate_base_covariance(self):

        # Go through one at a time
        n_base_covars = self.train_hr_aux_basecovariance.shape[0]

        per_upfactor_n_agg_tensor_k_ah_al = []
        per_upfactor_n_agg_tensor_k_al_al = []

        for u_index, u in enumerate(self.up_factors):
            self.hr_hw
            # retrieve lr_hw from self
            lr_hw = self.up_lr_hw[u_index]
            lr_indices_list = range(0, lr_hw**2) # covariance has square indices (pairwise)

            # includes 0 and last index and intevals are u * hr_hw wide
            lr_row_breaks = np.linspace(0, self.hr_hw**2, num = int(lr_hw + 1), dtype = int)

            # Placeholders
            n_agg_tensor_k_ah_al = torch.empty(size = (0, self.hr_hw**2, lr_hw**2))
            n_agg_tensor_k_al_al = torch.empty(size = (0, lr_hw**2, lr_hw**2))

            for n in range(0, n_base_covars):
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
                        lr_ranges_list.extend(range((j * u + (w * self.hr_hw) + row_base), (j * u + (w * self.hr_hw) + u + row_base)))

                    lrDict[i] = lr_ranges_list

                agg_tensor_k_ah_al = torch.empty(size = (self.hr_hw**2, lr_hw**2))
                agg_tensor_k_al_al = torch.empty(size = (lr_hw**2, lr_hw**2))

                # go column by column
                for col_ind in lr_indices_list:
                    # Only select n 
                    agg_tensor_k_ah_al[:, col_ind] = torch.mean((self.train_hr_aux_basecovariance[n, :, lrDict[col_ind]]), dim = 1)
    
                # Repeat on transposed output agg_tensor_k_ah_al
                for col_ind in lr_indices_list:
                    agg_tensor_k_al_al[:, col_ind] = torch.mean((torch.transpose(agg_tensor_k_ah_al, dim0 = 1, dim1 = 0)[:, lrDict[col_ind]]), dim = 1)

                # Concatinate
                n_agg_tensor_k_ah_al = torch.cat((n_agg_tensor_k_ah_al, agg_tensor_k_ah_al.unsqueeze(0)), dim = 0)
                n_agg_tensor_k_al_al = torch.cat((n_agg_tensor_k_al_al, agg_tensor_k_al_al.unsqueeze(0)), dim = 0)
            
            # Append to list of 
            per_upfactor_n_agg_tensor_k_ah_al.append(n_agg_tensor_k_ah_al)
            per_upfactor_n_agg_tensor_k_al_al.append(n_agg_tensor_k_al_al)

        return per_upfactor_n_agg_tensor_k_ah_al, per_upfactor_n_agg_tensor_k_al_al
    
    def predictive_distribution(self):
    #Calculation of the mean and variance of the Gaussian predictive distribution as well as of Log Marginal Likelihood. 
    #Here we implement the inversion through the Cholesky decomposition [torch.linalg.cholesky()] followed by the inversion [torch.cholesky_inverse(L)]. 
    #This is slighly different to the Algorithm 2.10 proposed in Rasmussen & Williams but more suitable to implement the weight correction performed by Reid et al.
        
        # Go through one at a time
        n_base_covars = self.train_hr_aux_basecovariance.shape[0]

        # empty tensor
        upfactors_n_mean = torch.empty(size = (0, n_base_covars, self.hr_hw, self.hr_hw ))
        upfactors_n_covariance = torch.empty(size = (0, n_base_covars, self.hr_hw**2, self.hr_hw**2))
        upfactors_n_lml = torch.empty(size = (0, n_base_covars))

        for u_index, u in enumerate(self.up_factors):
            n_mean = torch.empty(size = (0, self.hr_hw, self.hr_hw)) # mean field
            n_covariance = torch.empty(size = (0, self.hr_hw**2, self.hr_hw**2))
            n_lml = torch.empty(size = (0, ))

            for n in range(0, n_base_covars):
                # Calculate inverse via the Cholesky decomposition (Lower Triangular)
                L = torch.linalg.cholesky(self.per_upfactor_n_agg_tensor_k_al_al[u_index][n] + (torch.eye(n = self.per_upfactor_n_agg_tensor_k_al_al[u_index][n].shape[-1]) * self.gp_hyper_noise))
                k_inv = torch.cholesky_inverse(L)

                #### Mean ###
                pl_al_minus_mu = self.train_bed_lr[u_index][n] - torch.ones(size = self.train_bed_lr[u_index][n].shape) * self.gp_mu
                W = torch.matmul(self.per_upfactor_n_agg_tensor_k_ah_al[u_index][n], k_inv) 

                # Correction
                mean_flat = torch.div(torch.matmul(W, pl_al_minus_mu.reshape(-1).unsqueeze(1)), torch.matmul(W, torch.ones(size = (W.shape[-1], 1)))) + self.gp_mu
                # cast into 2D shape
                mean = mean_flat.reshape(1, int(np.sqrt(mean_flat.shape[0])), -1)

                ### Sigma ###
                # .mT transposes the last two dims of a matrix
                sigma = self.train_hr_aux_basecovariance[n] - torch.matmul(self.per_upfactor_n_agg_tensor_k_ah_al[u_index][n], torch.matmul(k_inv, self.per_upfactor_n_agg_tensor_k_ah_al[u_index][n].mT))

                ### LML ###
                # 2.30 in Rasmussen
                # https://d2l.ai/chapter_gaussian-processes/gp-inference.html
    
                # Term1: Kernel term
                term1 = torch.mul(torch.matmul(self.train_bed_lr[u_index][n].mT.reshape(1, -1), torch.matmul(k_inv, self.train_bed_lr[u_index][n].reshape(-1, 1))), 0.5)
    
                # Term2: Determinant term. 2 and 0.5 cancel each other out, sum in log space, log makes values negative
                term2 = torch.sum(torch.log(torch.diagonal(L)))
                # Need trick https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
                # Does not work: term2 = torch.mul(torch.log(torch.linalg.det(k_al_al + (torch.eye(n = k_al_al.shape[-1]) * noise))), 0.5).reshape(1, 1)
                # Flatten shape and extract n, natural log
    
                # Term3: Constant term: Only extracts shape from self.train_bed_lr
                term3 = (torch.log(torch.tensor(2 * math.pi)) * torch.tensor(self.train_bed_lr[u_index][n].reshape(-1).shape[0] * 0.5)).reshape((1, 1)) # n
                lml = - term1 - term2 - term3

                # Append term
                n_mean = torch.cat((n_mean, mean), dim = 0)
                n_covariance = torch.cat((n_covariance, sigma.unsqueeze(0)), dim = 0)
                n_lml = torch.cat((n_lml, lml.squeeze(1)), dim = 0)
            
            upfactors_n_mean = torch.cat((upfactors_n_mean, n_mean.unsqueeze(0)), dim = 0)
            upfactors_n_covariance = torch.cat((upfactors_n_covariance, n_covariance.unsqueeze(0)), dim = 0)
            upfactors_n_lml = torch.cat((upfactors_n_lml, n_lml.unsqueeze(0)), dim = 0)

        return upfactors_n_mean, upfactors_n_covariance, upfactors_n_lml
    