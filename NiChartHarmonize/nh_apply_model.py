import logging
import os
import pickle
import numpy as np
import pandas as pd
from statsmodels.gam.api import GLMGam, BSplines
import numpy.linalg as la
import copy
from typing import Union, Tuple

import pickle
#import dill

import logging
format='%(levelname)-8s [%(filename)s : %(lineno)d - %(funcName)20s()] %(message)s'
logging.basicConfig(level=logging.DEBUG, format = '\n' + format, datefmt='%Y-%m-%d:%H:%M:%S')
logger = logging.getLogger(__name__)

#logger.setLevel(logging.DEBUG)      ## While debugging
logger.setLevel(logging.INFO)    ## FIXME Comments will be removed in release version

#import sys
#pwd='/home/guray/Github/neuroHarmonizeV2/neuroHarmonizeV2'
#sys.path.append(pwd)

from .nh_utils import parse_init_data_and_model, make_dict_batches, make_design_dataframe_using_model, update_spline_vars_using_model, standardize_across_features_using_model, update_model_new_batch, fit_LS_model, find_parametric_adjustments, adjust_data_final, calc_aprior, calc_bprior, save_model, save_csv

#from nh_utils import fitLSModelAndFindPriorsV2

## FIXME Example to save curr vars
#print('SAVING')
#fname = 'sesstmp1.pkl'
#save_to_pickle(fname, [df_cov, batch_col, model])


def nh_harmonize_to_ref(model : Union[dict, str],
                        data : Union[pd.DataFrame, str],
                        covars : Union[pd.DataFrame, str],
                        ignore_saved_batch_params = False,
                        out_model : str = None,
                        out_csv : str = None
                        ) -> Tuple[dict, pd.DataFrame]:

    '''
    Harmonize each batch in the input dataset to the reference data (the input model):
    - Existing batches (in-sample): use saved parameters in the model;
    - New batches (out-of-sample): calculate parameters for the new batch and update the model
    
    Arguments
    ---------
    df_data (REQUIRED): Data to harmonize, in a pandas DataFrame 
        - Dimensions: n_samples x n_features
    
    df_cov (REQUIRED): Covariates, in a pandas DataFrame 
        - Dimensions: n_samples x n_covars;
        Columns in df_cov should match with harmonization covariates that were used for 
        calculating the input model

    ignore_saved_batch_params (OPTIONAL): Flag to ignore saved values for all batches and 
        recalculate them  (True or False, default = False)

    Returns
    -------
    model: Updated model. Parameters estimated for new batch(es) are added to the model_batches 
        dictionary
    
    df_out: Output dataframe with input covariates and harmonized variables
    '''
    
    ## FIXME : fixed seed here for now
    np.random.seed(11)

    ##################################################################
    ## Prepare data

    logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    logger.info('Running: nh_harmonize_to_ref()\n')    
    logger.info('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
    
    logger.info('------------------------------ Read Data -----------------------------')

    logger.info('  Reading model ...' + model)

    ## Read model file
    if isinstance(model, str):
        fmdl = open(model, 'rb')
        mdl_out = pickle.load(fmdl)
        fmdl.close()
    else:
        mdl_out = copy.deepcopy(model)
        
    mdl_batches = mdl_out['mdl_batches']
    mdl_ref = mdl_out['mdl_ref']
    batch_col = mdl_ref['dict_cov']['batch_col']
    
    ## Parse and check data
    logger.info('  Parsing / checking input data ...')    
    
    df_key, df_data, df_cov, dict_cov, dict_categories = parse_init_data_and_model(data, covars, mdl_ref)

    ##################################################################
    ## Harmonize each batch individually
    
    ## Output df
    df_h_data = pd.DataFrame(index = df_data.index, columns = df_data.columns)
    
    batch_col_vals = df_cov[batch_col].unique()
    for curr_batch in batch_col_vals:

        logger.info('Harmonizing batch : ' + str(curr_batch))
        
        ##################################################################
        ## Prepare data for the current batch
        
        ## Select batch
        ind_curr = df_cov[df_cov[batch_col] == curr_batch].index.tolist() 
                
        df_cov_curr = df_cov.loc[ind_curr, :].copy().reset_index(drop = True)
        df_data_curr = df_data.loc[ind_curr, :].copy().reset_index(drop = True)

        logger.info('  ------------------------------ Prep Data -----------------------------')
    
        ## Create dictionary with batch info
        logger.info('    Creating batches dictionary ...')
        dict_batches = make_dict_batches(df_cov_curr, batch_col)
        
        ## Create design dataframe
        logger.info('    Creating design matrix ...')            
        df_design, dict_design = make_design_dataframe_using_model(df_cov_curr, batch_col, mdl_ref)    

        ## Add spline terms to design dataframe
        logger.info('    Adding spline terms to design matrix ...')
        df_design, gam_formula = update_spline_vars_using_model(df_design, df_cov_curr, mdl_ref)

        ##################################################################
        ## COMBAT Step 1: Standardize dataset

        logger.info('  ------------------------------ COMBAT STEP 1 ------------------------------')
        
        logger.info('    Standardizing features ...')
        df_s_data, df_stand_mean = standardize_across_features_using_model(df_data_curr, df_design, mdl_ref)

        ##################################################################
        ##   ----------------  IN-SAMPLE HARMONIZATION ----------------
        ##   Current batch is one of the batches used for creating the ref model
        ##   - Skip Combat Step 2 (estimation of LS parameters)
        ##   - Use previously estimated model parameters to align new batch to reference model
        
        if curr_batch in mdl_batches['batch_values']:
            
            logger.info('  Batch in model; running IN-SAMPLE ...')

            ##################################################################
            ## COMBAT Step 3: Calculate harmonized data

            logger.info('  ------------------------------ COMBAT STEP 3 ------------------------------')
            
            ## Calculate harmonized data
            logger.info('    Adjusting final data ...\n')
            df_h_data_curr = adjust_data_final(df_s_data, mdl_batches['df_gamma_star'],
                                               mdl_batches['df_delta_star'],
                                               df_stand_mean, mdl_ref['df_pooled_stats'], dict_batches)
                        
        ##################################################################
        ##   ----------------  OUT-OF-SAMPLE HARMONIZATION ----------------
        ##   Current batch is not one of the batches used for creating the ref model
        ##   - Estimate parameters to align new batch to the reference model
        ##   - Save estimated parameters (update the model)
        ##   - Return harmonized data and updated model
        
        else:
            
            logger.info('  Batch not in model; running OUT-OF-SAMPLE ...')
            
            ##################################################################
            ### COMBAT Step 2: Calculate batch parameters (LS)

            logger.info('  ------------------------------ COMBAT STEP 2 ------------------------------')

            ##   Step 2.A : Estimate parameters
            logger.info('    Estimating location and scale (L/S) parameters ...')            
            dict_LS = fit_LS_model(df_s_data, df_design, dict_batches, mdl_ref['is_emp_bayes'])

            ##   Step 2.B : Adjust parameters    
            logger.info('    Adjusting location and scale (L/S) parameters ...')
            df_gamma_star, df_delta_star = find_parametric_adjustments(df_s_data, dict_LS,
                                                                       dict_batches, 
                                                                       mdl_ref['is_emp_bayes'])

            ##################################################################
            ## COMBAT Step 3: Calculate harmonized data

            logger.info('  ------------------------------ COMBAT STEP 3 ------------------------------')

            logger.info('    Adjusting final data ...\n')
            df_h_data_curr = adjust_data_final(df_s_data, df_gamma_star, df_delta_star, 
                                          df_stand_mean, mdl_ref['df_pooled_stats'], dict_batches)

            ###################################################################
            ## Update model
            
            logger.info('  ------------------------------ Update Model ------------------------------')

            logger.info('    Updating model with new batch ...\n')
            mdl_batches = update_model_new_batch(curr_batch, mdl_batches, df_gamma_star, df_delta_star)            

        ###################################################################
        ## Update harmonized data df

        logger.info('  ------------------------------ Update h_data  ------------------------------')
        df_h_data.loc[ind_curr, :] = df_h_data_curr.values
        
    ###################################################################
    ## Prepare output
    
    logger.info('------------------------------ Prepare Output ------------------------------')
    
    ## Set updated batches in output model (This should be the only change in the model)
    mdl_out['mdl_batches'] = mdl_batches
    
    ## Create out dataframe
    param_out_suff = '_HARM'    
    df_out = pd.concat([df_key, df_cov, df_h_data.add_suffix(param_out_suff)], axis=1)

    ###################################################################
    ## Return output
    if out_model is not None:
        logger.info('  Saving output model to:\n    ' + out_model)
        save_model(mdl_out, out_model)

    if out_csv is not None:
        logger.info('  Saving output data to:\n    ' + out_csv)
        save_csv(df_out, out_csv)

    logger.info('  Process completed \n')    

    return mdl_out, df_out
    
