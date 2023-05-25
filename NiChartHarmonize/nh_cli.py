from .nh_learn_model import nh_learn_ref_model
from .nh_apply_model import nh_harmonize_to_ref
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
    parser.add_argument("-i",
                        "--in_csv", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)

    # Model file
    help = "The input model file (only for action: apply)"
    parser.add_argument("-m",
                        "--in_model", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=False)

    # Key variable
    help = "Primary key of the data csv. If not provided, the first column is considered as the primary key"
    parser.add_argument("-k,",
                        "--key_var", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=False)

    # Batch variable
    help = "Batch variable (e.g. site, study, scanner)"
    parser.add_argument("-b",
                        "--batch_var", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=True)

    # Numeric variables
    help = "Numeric covariates that will be modeled using a linear model"
    parser.add_argument("-n",
                        "--num_vars", 
                        type=str,
                        #action='append', 
                        nargs='+',
                        help=help, 
                        default=[], 
                        required=False)

    # Categorical variables
    help = "Categoric covariates"
    parser.add_argument("-c",
                        "--cat_vars", 
                        type=str,
                        #action='append', 
                        nargs='+',
                        help=help, 
                        default=[], 
                        required=False)

    # Spline variables
    help = "Numeric covariates that will be modeled using a spline model"
    parser.add_argument("-s",
                        "--spline_vars", 
                        type=str,
                        #action='append', 
                        nargs='+',
                        help=help, 
                        default=[], 
                        required=False)
    
    # Variables to ignore
    help = "Variables that will be dropped / ignored"
    parser.add_argument("-g",
                        "--ignore_vars", 
                        type=str,
                        #action='append', 
                        nargs='+',
                        help=help, 
                        default=[], 
                        required=False)

    # Data variables
    help = "Variables that will be harmonized. If not provided, all variables in the input data csv after removing other listed covariates will be used as data variables"
    parser.add_argument("-d",
                        "--data_vars", 
                        type=str,
                        #action='append', 
                        nargs='+',
                        help=help, 
                        default=[], 
                        required=False)

    # Output file
    help = "The output model"
    parser.add_argument("-o",
                        "--out_model", 
                        type=str,
                        help=help, 
                        default=None, 
                        required=False)

    # Output data
    help = "The output data"
    parser.add_argument("-u",
                        "--out_csv", 
                        type=str,
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
        
    args = parser.parse_args()
    
    ### Print args
    #print(args)
    #print('aaa')
    #input()
    
    ## Call harmonize functions
    if args.action == 'learn':
        mdlOut, dfOut = nh_learn_ref_model(args.in_csv, 
                                           args.key_var, 
                                           args.batch_var,
                                           args.num_vars, 
                                           args.cat_vars, 
                                           args.spline_vars, 
                                           args.ignore_vars, 
                                           args.data_vars,                                            
                                           is_emp_bayes=True, 
                                           out_model = args.out_model, 
                                           out_csv = args.out_csv)
    if args.action == 'apply':
        mdlOut, dfOut = nh_harmonize_to_ref(args.in_csv, 
                                            args.in_model,
                                            out_model = args.out_model,
                                            out_csv = args.out_csv)
    
    return;
