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
    help = "Data file to be used as input"
    parser.add_argument("--data", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)
    
    # Covariates file
    help = "Covariates file to be used as input"
    parser.add_argument("--covars", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)

    # Model file
    help = "The model file"
    parser.add_argument("-m",
                        "--model", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)

    # Model file
    help = "The out data file"
    parser.add_argument("--out", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=False)

    # Batch var name
    help = "Batch field name used for harmonization"
    parser.add_argument("--batch_col", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)

    # Categorical vars name
    help = "Categorical field names"
    parser.add_argument("--cat_cols", 
                        type=str,
                        #action='append', 
                        nargs='+',
                        help=help, 
                        default=[], 
                        required=False)

    # Spline vars name
    help = "Field names to ignore"
    parser.add_argument("--spline_cols", 
                        type=str,
                        #action='append', 
                        nargs='+',
                        help=help, 
                        default=[], 
                        required=False)
    
    # Ignore vars name
    help = "Field names to ignore"
    parser.add_argument("--ignore_cols", 
                        type=str,
                        #action='append', 
                        nargs='+',
                        help=help, 
                        default=[], 
                        required=False)

    # Verbosity argument
    help = "Verbosity"
    parser.add_argument("-v", 
                        type=int, 
                        help=help, 
                        default=1, 
                        required=False)
        
    args = parser.parse_args()
    
    ## Print args
    print(args)
    print('aaa')
    input()
    
    ## Call harmonize functions  
    if args.action == 'learn':
        mdlOut, dfOut = nh_learn_ref_model(args.data, args.covars, args.batch_col, args.cat_cols,  
                                           args.ignore_cols, args.spline_cols)
    #if args.action == 'apply':
        #mdlOut, dfOut = nh_apply_model(args.data, args.covars, args.fbatch,
                                           #args.fspline, args.spline_min, args.spline_max)
    
    
    
    return
