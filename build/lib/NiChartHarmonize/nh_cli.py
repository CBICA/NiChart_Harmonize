from .nh_learn_model import nh_learn_ref_model
import argparse
import pandas as pd
from typing import Tuple, Union

def main():
    prog="neuroharm"
    description = "Harmonization reference data creation & harmonization of new data"
    parser = argparse.ArgumentParser(prog=prog,
                                     description=description)

    # Action argument
    help = "The action to be performed, either 'learn' or 'apply'"
    parser.add_argument("-a", 
                        "--action", 
                        type=str, 
                        help=help, 
                        default=None, 
                        required=True)
    
    # Data argument
    help = "The dataset to be used as input. Can be either a "\
            + "string filepath of a .csv file (str) or a pandas dataframe "\
            + "(pd.DataFrame)"
    parser.add_argument("--data", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)
    
    # Covariates data argument
    help = "The covariates to be used as input. Can be either a "\
            + "string filepath of a .csv file (str) or a pandas dataframe "\
            + "(pd.DataFrame)"
    parser.add_argument("--covars", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)

    # Model argument
    help = "The model to be used (only) for harmonization of new data. Can be either a "\
            + "string filepath of a .pkl.gz file or a tuple (dict, dict)."
    parser.add_argument("-m", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)
    
    # Verbosity argument
    help = "Verbosity"
    parser.add_argument("-v", 
                        type=int, 
                        help=help, 
                        default=1, 
                        required=False)
    
    # Save_path argument
    help = "Path to save the trained model. '.pkl.gz' file extension "\
            + "expected. If None is given, no model will be saved."
    parser.add_argument("-s", 
                        type=str, 
                        help=help, 
                        default='', 
                        required=False)
    
    args = parser.parse_args()
    print(args.data)
    input()
    
    ## Call harmonize functions
    df_data = pd.read_csv(args.data)
    df_cov = pd.read_csv(args.covars)
    
    if args.action == 'learn':
        mdlOut, dfOut = nh_learn_ref_model(df_data, df_cov, batch_col = 'Study',
                                                          spline_cols = ['Age_At_Visit'],
                                                          spline_bounds_min = [20], 
                                                          spline_bounds_max = [115])
    
    
    
    return
