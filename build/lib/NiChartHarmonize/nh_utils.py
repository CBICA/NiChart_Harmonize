import logging
import os
import pickle
import numpy as np
import pandas as pd
from statsmodels.gam.api import GLMGam, BSplines
import numpy.linalg as la
import copy
from statsmodels.gam.gam_cross_validation.cross_validators import KFold
import pickle
#import dill
import logging

## Set logging
format='%(levelname)-8s [%(filename)s : %(lineno)d - %(funcName)20s()] %(message)s'
logging.basicConfig(level=logging.DEBUG, format = '\n' + format, datefmt='%Y-%m-%d:%H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)      ## While debugging
#logger.setLevel(logging.INFO)    ## FIXME Comments will be removed in release version

## FIXME tmp
def save_to_pickle(fname, obj):
    out_file = open(fname, 'wb')
    pickle.dump(obj, out_file)
    out_file.close()

#####################################################################################
## Functions common to nh_learn_model and nh_apply_model

def make_dict_batches(df_cov, batch_col):
    '''
        Create a dictionary with meta data about batches  
    '''
    df_tmp = pd.get_dummies(df_cov[batch_col], prefix = batch_col)
    design_batch_indices = {}
    for bname in df_tmp.columns:
        design_batch_indices[bname] = df_tmp[df_tmp[bname] == 1].index.to_list()
    dict_batches = {'batch_values': df_cov[batch_col].unique().tolist(),
                    'design_batch_cols': df_tmp.columns.to_list(),
                    'n_batches': df_tmp.shape[1],
                    'n_samples': df_tmp.shape[0],
                    'n_samples_per_batch': np.array(df_tmp.sum(axis=0)),
                    'design_batch_indices': design_batch_indices}
    return dict_batches

def adjust_data_final(df_s_data, df_gamma_star, df_delta_star, df_stand_mean, 
                      df_pooled_stats, dict_batches):
    '''
        Apply estimated harmonization parameters
    '''

    ## Create output df
    df_h_data = df_s_data.copy()
    
    ## For each batch
    for i, b_tmp in enumerate(dict_batches['design_batch_cols']):
        
        ## Get sample index for the batch
        batch_idxs = dict_batches['design_batch_indices'][b_tmp]

        ## Get batch updated
        df_denom = np.sqrt(df_delta_star.loc[b_tmp,:].astype(np.float64))
        df_numer = df_s_data.loc[batch_idxs] - df_gamma_star.loc[b_tmp]
        df_h_data.loc[batch_idxs] = df_numer.div(df_denom)

    ## Get data updated
    df_vsq = np.sqrt(df_pooled_stats.loc['var_pooled'].astype(np.float64))
    df_h_data = df_h_data.multiply(df_vsq) + df_stand_mean

    return df_h_data


def fit_LS_model(df_s_data, df_design, dict_batches, is_emp_bayes=True):
    """
        Dataframe implementation of neuroCombat function fit_LS_model_and_find_priors
    """
    
    ## Get design matrix columns only for batch variables (batch columns)
    batch_cols = dict_batches['design_batch_cols']
    df_design_batch_only = df_design[batch_cols]
    
    ## Calculate gamma_hat
    df_tmp = df_design_batch_only.T.dot(df_design_batch_only)
    df_tmp.loc[:,:] = np.linalg.inv(np.matrix(df_tmp))    
    df_tmp = df_tmp.dot(df_design_batch_only.T)
    df_gamma_hat = df_tmp.dot(df_s_data)

    ## Calculate delta_hat
    df_delta_hat = pd.DataFrame(columns = df_gamma_hat.columns, index = df_gamma_hat.index)
    for i, b_tmp in enumerate(dict_batches['design_batch_cols']):
        batch_idxs = dict_batches['design_batch_indices'][b_tmp]
        df_delta_hat.loc[b_tmp, :] = df_s_data.loc[batch_idxs, :].var(axis = 0, ddof = 1)

    ## Calculate other params
    gamma_bar = None
    t2 = None
    a_prior = None
    b_prior = None
    if is_emp_bayes:
        df_gamma_bar = df_gamma_hat.mean(axis = 1) 
        df_t2 = df_gamma_hat.var(axis=1, ddof=1)
        df_a_prior = calc_aprior(df_delta_hat)
        df_b_prior = calc_bprior(df_delta_hat)

    ## Create out dict
    dict_LS = {'df_gamma_hat' : df_gamma_hat, 'df_delta_hat' : df_delta_hat, 
               'df_gamma_bar' : df_gamma_bar, 'df_t2' : df_t2, 
               'df_a_prior' : df_a_prior, 'df_b_prior' : df_b_prior}

    return dict_LS

def calc_aprior(df):
    m = df.mean(axis = 1)
    s2 = df.var(axis = 1, ddof = 1)
    df_out = (2 * s2 +m**2) / s2 
    return df_out 

def calc_bprior(df):
    m = df.mean(axis = 1)
    s2 = df.var(axis = 1, ddof = 1)
    df_out = ( m * s2 + m * m * m ) / s2
    return df_out 

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0
    while change > conv:
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - np.dot(g_new.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(axis=1)
        d_new = postvar(sum2, n, a, b)

        change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
        g_old = g_new #.copy()
        d_old = d_new #.copy()
        count = count + 1
    adjust = (g_new, d_new)
    return adjust 

def find_parametric_adjustments(df_s_data, dict_LS, dict_batches, is_emp_bayes):
    '''
        Calculate adjusted gamma and delta values (gamma and delta star)
    '''

    ## Get data into numpy matrices
    s_data = np.array(df_s_data).T    

    if is_emp_bayes == False:
        df_gamma_star = dict_LS['df_gamma_hat']
        delta_star = dict_LS['df_delta_hat']
    else:
        design_batch_indices = dict_batches['design_batch_indices'] 

        gamma_star, delta_star = [], []
        for i, b_tmp in enumerate(dict_batches['design_batch_cols']):
            
            ## Get data
            batch_idxs = dict_batches['design_batch_indices'][b_tmp]
            s_data = np.array(df_s_data.loc[batch_idxs, :]).T
            gamma_hat = np.array(dict_LS['df_gamma_hat'].loc[b_tmp, :])
            delta_hat = np.array(dict_LS['df_delta_hat'].loc[b_tmp, :])
            gamma_bar = np.array(dict_LS['df_gamma_bar'].loc[b_tmp])
            t2 = np.array(dict_LS['df_t2'].loc[b_tmp])
            a_prior = np.array(dict_LS['df_a_prior'].loc[b_tmp])
            b_prior = np.array(dict_LS['df_b_prior'].loc[b_tmp])
            
            ## Calculate parameters
            temp = it_sol(s_data, gamma_hat, delta_hat, gamma_bar, t2, a_prior, b_prior)
            
            
            gamma_star.append(temp[0])
            delta_star.append(temp[1])
                        
        gamma_star = np.array(gamma_star)
        delta_star = np.array(delta_star)
        
    df_gamma_star = pd.DataFrame(data = gamma_star, index = dict_batches['design_batch_cols'], 
                                 columns = df_s_data.columns.tolist())
    df_delta_star = pd.DataFrame(data = delta_star, index = dict_batches['design_batch_cols'], 
                                 columns = df_s_data.columns.tolist())

    return df_gamma_star, df_delta_star

def save_results(results, out_file_name):
    """
    Save harmonization results as a pickle file
    """
    out_file_name_full = os.path.abspath(out_file_name)
    out_dir = os.path.dirname(out_file_name_full)

    if os.path.exists(out_dir) == False:
        os.makedirs(out_dir)

    if os.path.exists(out_file_name_full):
        raise ValueError('Out file already exists: %s. Change name or delete to save.' % out_file_name_full)

    out_file = open(out_file_name, 'wb')
    pickle.dump(results, out_file)
    out_file.close()

#####################################################################################
## Functions specific to LearnRefModel
def parse_init_data(df_data, df_cov, batch_col, categoric_cols, spline_cols, spline_bounds_min, spline_bounds_max,
                    ignore_cols, out_file_name):
    """ 
    Read initial data, verify it, and extract meta data about variables
    """
    
    ## Check output file
    if out_file_name != None:
        out_file_name_full = os.path.abspath(out_file_name)
        if os.path.exists(out_file_name_full):
            raise ValueError('Out file already exists: %s. Change name or delete to save.' % out_file_name)
    
    ## Verify data dataframe
    if not isinstance(df_data, pd.DataFrame):
        raise ValueError('Data must be a pandas dataframe')

    ## Verify covar dataframe
    if not isinstance(df_cov, pd.DataFrame):
        raise ValueError('Covars must be a pandas dataframe')

    ## Reset index for data and covar dataframes
    df_data = df_data.reset_index(drop = True)
    df_cov = df_cov.reset_index(drop = True)

    ## Verify input columns exist
    cols_combined = [batch_col] + categoric_cols + spline_cols
    
    for tmp_col in cols_combined:
        if tmp_col not in df_cov.columns:
            raise ValueError('Variable not found in covariates dataframe: ' + tmp_col)

    ## Verify data and covar have the same number of samples
    if df_data.shape[0] != df_cov.shape[0]:
        raise ValueError('Data and covariables files must have the same number of rows')

    ## Replace special characters in batch values
    ## FIXME Special characters fail in GAM formula
    ##   Update this using patsy quoting in next version
    df_cov[batch_col] = df_cov[batch_col].str.replace('-', '_').str.replace(' ', '_')
    df_cov[batch_col] = df_cov[batch_col].str.replace('.', '_').str.replace('/', '_')
    
    ## FIXME Remove null values in data dataframe (TODO)
    
    ## FIXME Remove null values in covar dataframe (TODO)
    
    ## Remove columns that should be ignored
    df_data = df_data.drop(ignore_cols, axis=1, errors='ignore')

    ## Detect numeric columns
    numeric_cols = df_cov.columns[df_cov.columns.isin(cols_combined) == False].tolist()
    covar_cols_all = numeric_cols + categoric_cols + spline_cols

    ## Create dictionary of covars
    dict_cov = {'covar_cols_all' : covar_cols_all, 'batch_col' : batch_col, 
                'numeric_cols' : numeric_cols, 'categoric_cols' : categoric_cols, 
                'spline_cols' : spline_cols, 'spline_bounds_min' : spline_bounds_min,
                'spline_bounds_max' : spline_bounds_max, 'ignore_cols': ignore_cols}

    ## Create dictionary of categorical covars
    dict_categories ={}
    for tmp_var in categoric_cols:
        categoric_vals = df_cov[tmp_var].unique().tolist()
        dict_categories[tmp_var] = categoric_vals 

    return df_data, df_cov, dict_cov, dict_categories


def make_design_dataframe(df_cov, dict_cov):
    """
    Expand the covariates dataframe adding columns that will constitute the design matrix
    New columns in the output dataframe are: 
        - one-hot matrix of batch variables (full)
        - one-hot matrix for each categorical var (removing the first column)
        - column for each continuous_cols
        - spline variables are skipped (added later in a separate function)
    """
    ## Make output dataframe
    df_design_out = df_cov.copy()
    
    ## Make output dict
    dict_design = {}
    
    ## Keep columns that will be included in the final design matrix
    design_vars = []

    ## Add one-hot encoding of batch variable
    df_tmp = pd.get_dummies(df_design_out[dict_cov['batch_col']], prefix = dict_cov['batch_col'], dtype = float)
    design_batch_cols = df_tmp.columns.tolist()
    df_design_out = pd.concat([df_design_out, df_tmp], axis = 1)
    design_vars = design_vars + design_batch_cols

    ## Add numeric variables
    ##   Numeric variables do not need any manipulation; just add them to the list of design variables
    numeric_cols = dict_cov['numeric_cols']
    design_vars = design_vars + numeric_cols
    dict_design['numeric_cols'] = numeric_cols

    ## Add one-hot encoding for each categoric variable
    categoric_cols = []
    for tmp_var in dict_cov['categoric_cols']:
        df_tmp = pd.get_dummies(df_design_out[tmp_var], prefix = tmp_var, drop_first = True, 
                                dtype = float)
        df_design_out = pd.concat([df_design_out, df_tmp], axis = 1)
        categoric_cols = categoric_cols + df_tmp.columns.tolist()
    design_vars = design_vars + categoric_cols
    dict_design['categoric_cols'] = categoric_cols
    
    ## Add dict item for non-batch columns
    dict_design['non_batch_cols'] = numeric_cols + categoric_cols

    ## Return output vars
    return df_design_out[design_vars], dict_design

def calc_spline_model(df_spline, spline_bounds_min = None, spline_bounds_max = None, param_spline_doff = 10, 
                      param_spline_degree = 3):
    '''
    calculate spline model for the selected spline variables
    '''
    
    doff = [param_spline_doff] * df_spline.shape[1] 
    degree = [param_spline_degree] * df_spline.shape[1] 

    ## If spline_bounds are provided construct the knot_kwds arg
    ## knot_kwds arg for multiple variables is a "list of dictionaries" with min/max bound values
    if len(spline_bounds_min) > 0:        
        kwds = []
        for i, tmp_min in enumerate(spline_bounds_min):
            kwds = kwds + [{'lower_bound' : tmp_min, 'upper_bound' : spline_bounds_max[i]}]
        bsplines = BSplines(x = df_spline, df = doff, degree = degree , knot_kwds = kwds)
    else:
        bsplines = BSplines(x = df_spline, df = doff, degree = degree)

    ## Return spline model
    return bsplines

def add_spline_vars(df_design, dict_design, df_cov, dict_cov):
    """
    Add columns for spline variables to design dataframe
        - spline basis columns for each spline var (based on Ray's implementation)
    """
    
    ## Make output dataframe
    df_design_out = df_design.copy()
    
    ## Keep columns that will be included in the final design matrix
    design_vars = df_design.columns.tolist()

    ## Get dict vars
    spline_cols = dict_cov['spline_cols']
    spline_bounds_min = dict_cov['spline_bounds_min']
    spline_bounds_max = dict_cov['spline_bounds_max']

    ## Add spline basis for spline variables
    bsplines = None
    gam_formula = None
    if len(spline_cols) > 0:
        bsplines = calc_spline_model(df_cov[spline_cols], spline_bounds_min, spline_bounds_max)
        df_bsplines = pd.DataFrame(data = bsplines.basis, columns = bsplines.col_names)

        ## Used for fitting the gam model on data with bsplines
        ## Note: Variable names are quoted to allow characters such as ' ' and '-' in var names

        ## FIXME handle this using patsy quoting 
        #gam_formula = 'y ~ ' + ' + '.join(['"' + x + '"' for x in design_vars]) + ' - 1 '

        gam_formula = 'y ~ ' + ' + '.join(design_vars) + ' - 1 '


        df_design_out = pd.concat([df_design_out, df_bsplines], axis = 1)
        design_vars = design_vars + bsplines.col_names
        
        ## Add spline meta data to dict
        dict_design['spline_cols'] = bsplines.col_names
        
        ## Add spline cols to list of non-batch cols
        dict_design['non_batch_cols'] = dict_design['non_batch_cols'] + bsplines.col_names

    ## Return output vars
    return df_design_out[design_vars], dict_design, bsplines, gam_formula

def calc_B_hat(df_data, df_design, dict_batches, bsplines = None, gam_formula = None):
    '''
    Calculate the B hat values
    '''

    #save_to_pickle('/home/guray/sesstmp3.pkl', [df_data, df_design, dict_batches, bsplines, gam_formula])

    ## During estimation print a dot in every "param_dot_count" variables to show the progress
    param_dot_count = 5
    
    ## Get data array
    np_data = np.array(df_data, dtype='float32').T

    ## Get batch info
    n_batch = dict_batches['n_batches']
    n_sample = dict_batches['n_samples']
    sample_per_batch = dict_batches['n_samples_per_batch']

    ## Perform smoothing with GAMs if specified
    np_design = np.array(df_design, dtype='float32')
    
    if bsplines == None:
        B_hat = np.dot(np.dot(np.linalg.inv(np.dot(np_design.T, np_design)), np_design.T), np_data.T)

    else:
        if df_data.shape[1] > 10:
            logger.info('Smoothing more than 10 variables may take several minutes of computation.')

        # initialize penalization weight (not the final weight)
        alpha = np.array([1.0] * bsplines.k_variables)
        
        # initialize an empty matrix for beta
        B_hat = np.zeros((df_design.shape[1], df_data.shape[1]))
        
        # Estimate beta for each variable to be harmonized
        logger.info('Estimating gam model for variables')
        print('   Printing a dot for every variable computed: ')
        df_design_extended = df_design.copy()
        for i, y_col in enumerate(df_data.columns.to_list()):
            
            ## Show progress
            print('.', end = '', flush = True)
            if np.mod(i, param_dot_count) == param_dot_count - 1:
                print('')
            
            ## Add single var to design matrix
            df_design_extended.loc[:, 'y'] = df_data[y_col]
            
            ## Estimate gam model
            gam_bsplines = GLMGam.from_formula(gam_formula, data = df_design_extended, smoother = bsplines, 
                                               alpha = alpha)
            res_bsplines = gam_bsplines.fit()
            
            # Optimal penalization weights alpha can be obtained through gcv/kfold
            # Note: kfold is faster, gcv is more robust

            #gam_bsplines.alpha = gam_bsplines.select_penweight_kfold()[0]

            ## FIXME 
            ##  By default, select_penweight_kfold uses a kfold object with shuffle = True
            ##  This makes it give non-deterministic output
            ##  Current fix :  Call it using an explicit kfold object without shuffling
            ##  To do :  Shuffle data initially once 
            kf = KFold(k_folds = 5, shuffle = False)
            gam_bsplines.alpha = gam_bsplines.select_penweight_kfold(cv_iterator = kf)[0]

            res_bsplines_optim = gam_bsplines.fit()
            
            B_hat[:, i] = res_bsplines_optim.params
        print('\n')

    ## Create B hat dataframe
    df_B_hat = pd.DataFrame(data = B_hat, columns = df_data.columns.tolist(), 
                            index = df_design.columns.tolist())
    
    ## Return B hat dataframe
    return df_B_hat


def standardize_across_features(df_data, df_design, df_B_hat, dict_design, dict_batches):
    """
    The original neuroCombat function standardize_across_features plus
    necessary modifications.
    
    This function will return all estimated parameters in addition to the
    standardized data.
    """

    ## Get design columns for batches and non-batches    
    bcol = dict_batches['design_batch_cols']
    nbcol = dict_design['non_batch_cols']
    
    ## Calculate ratio of samples in each batch (to total number of samples)
    r_batches = df_design[bcol].sum() / df_design.shape[0]

    ## Grand mean is the sum of b_hat values for batches weighted by ratio of samples 
    df_grand_mean = df_B_hat.loc[bcol, :].multiply(r_batches, axis = 0).sum(axis = 0)

    ## Data regressed to beta's "for all covars" in the design matrix
    df_fit_all = pd.DataFrame(data = np.dot(df_design, df_B_hat), 
                              columns = df_B_hat.columns, index = df_design.index)

    ## Data regressed to beta's "for non-batch covars" in the design matrix
    df_fit_nb = pd.DataFrame(data = np.dot(df_design[nbcol], df_B_hat.loc[nbcol, :]), 
                             columns = df_B_hat.columns, index = df_design.index)
    
    ## Residuals from the fit
    df_res = df_data - df_fit_all

    ## Calculate pooled var 
    df_var_pooled = (df_res ** 2).mean(axis = 0)

    ## Calculate stand var
    df_stand_mean = df_grand_mean + df_fit_nb
    
    ## Calculate s_data
    df_s_data = (df_data - df_stand_mean).div(np.sqrt(df_var_pooled), axis = 1)

    ### Keep grand_mean and var_pooled in a single dataframe
    df_pooled_stats = pd.concat([df_grand_mean, df_var_pooled], axis = 1).T
    df_pooled_stats.index = ['grand_mean', 'var_pooled']

    return df_s_data, df_stand_mean, df_pooled_stats

#####################################################################################
## Functions specific to HarmonizeToRef

def parse_init_data_and_model(df_data, df_cov, out_file_name, mdl):
    '''
        Verify that init data matches the model
        Extract dictionaries for variables and batches
    '''
    
    ## Check output file
    if out_file_name != None:
        out_file_name_full = os.path.abspath(out_file_name)
        if os.path.exists(out_file_name_full):
            raise ValueError('Out file already exists: %s. Change name or delete to save.' % out_file_name)
    
    ## Verify data dataframe
    if not isinstance(df_data, pd.DataFrame):
        raise ValueError('Data must be a pandas dataframe')

    ## Verify covar dataframe
    if not isinstance(df_cov, pd.DataFrame):
        raise ValueError('Covars must be a pandas dataframe')

    ## Reset index for data and covar dataframes
    df_data = df_data.reset_index(drop = True)
    df_cov = df_cov.reset_index(drop = True)

    ## Read covars from the model
    dict_cov = mdl['dict_cov']
    
    ## Read batch col from the model
    batch_col = dict_cov['batch_col']

    ## Read categories from the model
    dict_categories = mdl['dict_categories']

    ## Verify that input columns exist
    cols_combined = [batch_col] + dict_cov['covar_cols_all']

    for tmp_col in cols_combined:
        if tmp_col not in df_cov.columns:
            raise ValueError('Variable not found in covariates dataframe: ' + tmp_col)

    ## Verify that data and covar have the same number of samples
    if df_data.shape[0] != df_cov.shape[0]:
        raise ValueError('Data and covar files must have the same number of rows')

    ## Replace special characters in batch values
    ## FIXME Special characters fail in GAM formula
    ##   Update this using patsy quoting in next version
    df_cov[batch_col] = df_cov[batch_col].str.replace('-', '_').str.replace(' ', '_')
    df_cov[batch_col] = df_cov[batch_col].str.replace('.', '_').str.replace('/', '_')
    
    ## FIXME Remove null values in data dataframe (TODO)

    ## FIXME Remove null values in covar dataframe (TODO)
    
    ## Remove columns that should be ignored
    df_data = df_data.drop(dict_cov['ignore_cols'], axis=1, errors='ignore')

    ## Remove rows with categorical values not present in model data
    num_sample_init = df_cov.shape[0]
    for tmp_col in dict_cov['categoric_cols']:         ## For each cat var from the dict_cov that was saved in model
        tmp_vals = dict_categories[tmp_col]         ##   Read values of the cat var
        df_cov = df_cov[df_cov[tmp_col].isin(tmp_vals)]  ##   Remove rows with values different than those in model
    
    df_data = df_data.loc[df_cov.index]           ## Remove deleted rows from the data dataframe
    num_sample_new = df_cov.shape[0]
    num_diff = num_sample_init - num_sample_new
    if num_diff != 0:
        logger.info('WARNING: Samples with categorical values not in model data are discarded ' + 
                    ', n removed = ' + str(num_diff))

    return df_data, df_cov, dict_cov, dict_categories

def make_design_dataframe_using_model(df_cov, batch_col, mdl):
    '''
        Expand the covariates dataframe adding columns that will constitute the design matrix
        New columns in the output dataframe are: 
            - one-hot matrix of batch variables (full)
            - one-hot matrix for each categorical var (removing the first column)
            - column for each continuous_cols
            - spline variables are skipped (added later in a separate function)
    '''
    categoric_cols = mdl['dict_cov']['categoric_cols']
    spline_cols = mdl['dict_cov']['spline_cols']    
    mdl_dict_design = mdl['dict_design']
    mdl_dict_categories = mdl['dict_categories']
    
    ## Make output dataframe
    df_design_out = df_cov.copy()
    
    ## Make output dict
    dict_design = copy.deepcopy(mdl_dict_design)
            
    ## Keep columns that will be included in the final design matrix
    design_vars = []

    ## Add one-hot encoding of batch variable
    df_tmp = pd.get_dummies(df_design_out[batch_col], prefix = batch_col, dtype = float)
    design_batch_cols = df_tmp.columns.tolist()
    df_design_out = pd.concat([df_design_out, df_tmp], axis = 1)
    design_vars = design_vars + design_batch_cols

    ## Add numeric variables (from mdl dict_design)
    design_vars = design_vars + mdl_dict_design['numeric_cols']

    ## Add one-hot encoding for each categoric variable
    ##   If category values match the ones from the mdl, it's simple dummy encoding
    ##   If not: 
    ##     - extend the df with a tmp df with all category values from the mdl;
    ##     - apply dummy encoding
    ##     - remove the tmp df
    for tmp_var in categoric_cols:
        
        ## Compare cat values from the mdl to values for curr data
        mdl_tmp_vals = mdl_dict_categories[tmp_var]
        tmp_vals = df_design_out[tmp_var].sort_values().unique().tolist()
        
        if tmp_vals == mdl_tmp_vals:        
            df_tmp = pd.get_dummies(df_design_out[tmp_var], prefix = tmp_var, drop_first = True, dtype = float)
        
        else:
            ## Create a tmp df with cat values from mdl
            df_tmp = pd.DataFrame(data = mdl_tmp_vals, columns = [ tmp_var ])
            
            ## Combine it to cat values for the curr data
            df_tmp = pd.concat([df_tmp, df_design_out[[tmp_var]]])
            
            ## Create dummy cat vars
            df_tmp = pd.get_dummies(df_tmp[tmp_var], prefix = tmp_var, drop_first = True,
                                    dtype = float) 
            
            ## Remove the tmp df values
            df_tmp = df_tmp.iloc[len(mdl_tmp_vals):,:]
            
        df_design_out = pd.concat([df_design_out, df_tmp], axis = 1)
        
        design_vars = design_vars + df_tmp.columns.tolist()

    ## Return output vars
    return df_design_out[design_vars], dict_design
    
def update_spline_vars_using_model(df_design, df_cov, mdl):
    '''
        Add columns for spline variables to design dataframe using prev spline mdl
            - spline basis columns for each spline var (based on Ray's implementation)
    '''
    
    ## Read mdl spline vars
    spline_cols = mdl['dict_cov']['spline_cols']
    bsplines = mdl['bsplines']
    
    ## Make output dataframe
    df_design_out = df_design.copy()

    ## Keep columns that will be included in the final design matrix
    design_vars = df_design.columns.tolist()

    ## Add spline basis for spline variables
    gam_formula = None
    if len(spline_cols) > 0:
        
        ## Existing bspline basis are used to calculate spline columns for the new data
        np_cov_spline = np.array(df_cov[spline_cols])

        ## FIXME handle this using patsy quoting 
        #gam_formula = 'y ~ ' + ' + '.join(['"' + x + '"' for x in design_vars]) + ' - 1 '

        ## Used for fitting the gam mdl on data with bsplines
        gam_formula = 'y ~ ' + ' + '.join(design_vars) + ' - 1 '

        bsplines_basis = bsplines.transform(np_cov_spline)
        df_bsplines = pd.DataFrame(data = bsplines_basis, columns = bsplines.col_names)

        df_design_out = pd.concat([df_design_out, df_bsplines], axis = 1)
        
        design_vars = design_vars + bsplines.col_names

    ## Return output vars
    return df_design_out[design_vars], gam_formula

def standardize_across_features_using_model(df_data, df_design, mdl):
    """
    The original neuroCombat function standardize_across_features plus
    necessary modifications.
    
    This function will apply a pre-trained harmonization mdl to new data.
    """
    
    ## Get mdl data
    mdl_nbcol = mdl['dict_design']['non_batch_cols']
    mdl_var_pooled = mdl['df_pooled_stats'].loc['var_pooled', :]
    mdl_grand_mean = mdl['df_pooled_stats'].loc['grand_mean', :]
    mdl_B_hat = mdl['df_B_hat']

    ## Data regressed to beta's "for non-batch covars" in the design matrix
    ##   Calculated B_hat values from the model
    df_fit_nb = pd.DataFrame(data = np.dot(df_design[mdl_nbcol], mdl_B_hat.loc[mdl_nbcol, :]), 
                             columns = mdl_B_hat.columns, index = df_design.index)

    ## Calculate stand var
    df_stand_mean = mdl_grand_mean + df_fit_nb
    
    ## Calculate s_data
    df_s_data = (df_data - df_stand_mean).div(np.sqrt(mdl_var_pooled), axis = 1)

    return df_s_data, df_stand_mean

def update_model_new_batch(new_batch, mdl_batches, df_gamma_star, df_delta_star):
    '''
    Add estimated batch parameters to the model
    '''
    mdl_batches['batch_values'] = mdl_batches['batch_values'] + [new_batch]
    mdl_batches['df_gamma_star'] = pd.concat([mdl_batches['df_gamma_star'], df_gamma_star], axis=0)
    mdl_batches['df_delta_star'] = pd.concat([mdl_batches['df_delta_star'], df_delta_star], axis=0)
    
    return mdl_batches
