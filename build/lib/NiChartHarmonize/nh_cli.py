from .nh_learn_model import nh_learn_ref_model
import argparse
import pandas as pd
from typing import Tuple, Union

def main():
    prog="neuroharm"
    description = "Harmonization learn ref model & apply to new data"
    parser = argparse.ArgumentParser(prog=prog,
                                     description=description)

    # Action
    help = "The action to be performed, either 'learn' or 'apply'"
    parser.add_argument("-a", 
                        "--action", 
                        type=str, 
                        help=help, 
                        default=None, 
                        required=True)
    
    # Data file
    help = "The dataset to be used as input. Can be either a "\
            + "string filepath of a .csv file (str) or a pandas dataframe "\
            + "(pd.DataFrame)"
    parser.add_argument("--data", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)
    
    # Covariates file
    help = "The covariates to be used as input. Can be either a "\
            + "string filepath of a .csv file (str) or a pandas dataframe "\
            + "(pd.DataFrame)"
    parser.add_argument("--covars", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)

    # Model file
    help = "The model to use for harmonization (only for action apply). Can be either a "\
            + "string filepath of a .pkl.gz file or a tuple (dict, dict)."
    parser.add_argument("-m", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)

    # Key var name
    help = "The primary key field that links data and covariates"
    parser.add_argument("--fkey", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=False)

    # Batch var name
    help = "Batch field name used for harmonization"
    parser.add_argument("--fbatch", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)

    # Categorical vars name
    help = "Categorical field names"
    parser.add_argument("--fcat", 
                        type=str,
                        action='append', 
                        nargs='+',
                        help=help, 
                        default=None, 
                        required=False)

    # Spline vars name
    help = "Field names to ignore"
    parser.add_argument("--fignore", 
                        type=str,
                        action='append', 
                        nargs='+',
                        help=help, 
                        default=None, 
                        required=False)

    # Ignore vars name
    help = "Field names to ignore"
    parser.add_argument("--fspline", 
                        type=str,
                        action='append', 
                        nargs='+',
                        help=help, 
                        default=None, 
                        required=False)
    
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
    print(args)
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
