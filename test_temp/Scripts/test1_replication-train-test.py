import pandas as pd
import numpy as np
import pickle
import sys
import os

#BDIR = '/home/guray/'
#if os.path.exists(BDIR) == False:
    #BDIR = '/home/guraylab/AIBIL'

#pwd = BDIR + '/Github/NiChartHarmonize/NiChartHarmonize'
#sys.path.append(pwd)
#import nh_learn_model as nhlm
#import nh_apply_model as nham

from NiChartHarmonize import nh_learn_model as nhlm
from NiChartHarmonize import nh_apply_model as nham


#######################################################################################
print('\nPreparing data')

## Read dataset A (ADNI+BLSA+UKBB)              --> TRAIN
dfROI_A = pd.read_csv('../Data/s2_Set1_MUSE.csv')
dfCOV_A = pd.read_csv('../Data/s2_Set1_COV.csv')

## Create dataset B (Only UKBB from set A)      --> REAPPLY USING SAME TRAINING BATCH
df_A = pd.concat([dfROI_A, dfCOV_A], axis=1)
df_A = df_A[df_A.SITE=='UKBIOBANK']
dfROI_B = df_A[dfROI_A.columns]
dfCOV_B = df_A[dfCOV_A.columns]

## Create dataset C (Only BLSA from set A)      --> REAPPLY USING SAME TRAINING BATCH
df_A = pd.concat([dfROI_A, dfCOV_A], axis=1)
df_A = df_A[df_A.SITE=='BLSA-3T']
dfROI_C = df_A[dfROI_A.columns]
dfCOV_C = df_A[dfCOV_A.columns]

## Create dataset D (Only BLSA from set A, site name renamed to BLSATMP)  --> OUT OF SAMPLE
dfROI_D = dfROI_C.copy()
dfCOV_D = dfCOV_C.copy()
dfCOV_D.loc[dfCOV_D.SITE == 'BLSA-3T', 'SITE'] = 'BLSATMP'

## Read dataset E (OASIS+SHIP)                  --> OUT OF SAMPLE
dfROI_E = pd.read_csv('../Data/s2_Set4_MUSE.csv')
dfCOV_E = pd.read_csv('../Data/s2_Set4_COV.csv')


## Create dataset F (OASIS+SHIP)                --> ALL TOGETHER
dfROI_F = pd.concat([dfROI_B, dfROI_C, dfROI_D, dfROI_E]).reset_index()
dfCOV_F = pd.concat([dfCOV_B, dfCOV_C, dfCOV_D, dfCOV_E]).reset_index()
indTmp = np.array(dfROI_F.index)
np.random.shuffle(indTmp)                           
dfROI_F = dfROI_F.loc[indTmp, dfROI_A.columns]
dfCOV_F = dfCOV_F.loc[indTmp, dfCOV_A.columns]

#######################################################################################
print('\nRunning models')
print('\nSet A')
input('Learn ref model.\nPress a key')
mdlA, outA = nhlm.nh_learn_ref_model(dfROI_A, dfCOV_A, batch_col = 'SITE', 
                                     spline_cols = ['Age'], spline_bounds_min = [20], spline_bounds_max = [95])

print('\nSet B')
input('Harmonize data to model.\nPress a key')
mdltmp = mdlA.copy()
mdlB, outB = nham.nh_harmonize_to_ref(mdltmp, dfROI_B, dfCOV_B)

print('\nSet C')
input('Harmonize data to model.\nPress a key')
mdltmp = mdlA.copy()
mdlC, outC = nham.nh_harmonize_to_ref(mdltmp, dfROI_C, dfCOV_C)

print('\nSet D')
input('Harmonize data to model.\nPress a key')
mdltmp = mdlA.copy()
mdlD, outD = nham.nh_harmonize_to_ref(mdltmp, dfROI_D, dfCOV_D)

print('\nSet E')
input('Harmonize data to model.\nPress a key')
mdltmp = mdlA.copy()
mdlE, outE = nham.nh_harmonize_to_ref(mdltmp, dfROI_E, dfCOV_E)

print('\nSet F')
input('Harmonize data to model.\nPress a key')
mdltmp = mdlA.copy()
mdlF, outF = nham.nh_harmonize_to_ref(mdltmp, dfROI_F, dfCOV_F)

#######################################################################################
print('\nDisplaying outputs')

print('\n              Estimated parameters for the ref model (mdl["mdl_ref"])')
input('Press a key ...')
print('\n mdl_ref')
print(mdlA['mdl_ref'].keys())
print('\n dict_cov')
print(mdlA['mdl_ref']['dict_cov'])
print('\n dict_categories')
print(mdlA['mdl_ref']['dict_categories'])
print('\n dict_design')
print(mdlA['mdl_ref']['dict_design'])
print('\n df_B_hat')
print(mdlA['mdl_ref']['df_B_hat'])
print('\n df_pooled_stats')
print(mdlA['mdl_ref']['df_pooled_stats'])
print('\n')


print('\n              Estimated batch parameters (mdl["mdl_batches"]) - batch values  ')
input('Press a key ...')
print('\nMDL_A:')
print(mdlA['mdl_batches']['batch_values'])
print('\nMDL_B:')
print(mdlB['mdl_batches']['batch_values'])
print('\nMDL_C:')
print(mdlC['mdl_batches']['batch_values'])
print('\nMDL_D:')
print(mdlD['mdl_batches']['batch_values'])
print('\nMDL_E:')
print(mdlE['mdl_batches']['batch_values'])
print('\nMDL_F:')
print(mdlF['mdl_batches']['batch_values'])

print('\n              Estimated batch parameters (mdl["mdl_batches"]) - gamma star values  ')
input('Press a key ...')
print('\nMDL_A:')
print(mdlA['mdl_batches']['df_gamma_star'].mean(axis=1))
print('\nMDL_B:')
print(mdlB['mdl_batches']['df_gamma_star'].mean(axis=1))
print('\nMDL_C:')
print(mdlC['mdl_batches']['df_gamma_star'].mean(axis=1))
print('\nMDL_D:')
print(mdlD['mdl_batches']['df_gamma_star'].mean(axis=1))
print('\nMDL_E:')
print(mdlE['mdl_batches']['df_gamma_star'].mean(axis=1))
print('\nMDL_F:')
print(mdlF['mdl_batches']['df_gamma_star'].mean(axis=1))


print('\n              Estimated batch parameters (mdl["mdl_batches"]) - delta star values  ')
input('Press a key ...')
print('\nMDL_A:')
print(mdlA['mdl_batches']['df_delta_star'].mean(axis=1))
print('\nMDL_B:')
print(mdlB['mdl_batches']['df_delta_star'].mean(axis=1))
print('\nMDL_C:')
print(mdlC['mdl_batches']['df_delta_star'].mean(axis=1))
print('\nMDL_D:')
print(mdlD['mdl_batches']['df_delta_star'].mean(axis=1))
print('\nMDL_E:')
print(mdlE['mdl_batches']['df_delta_star'].mean(axis=1))
print('\nMDL_F:')
print(mdlF['mdl_batches']['df_delta_star'].mean(axis=1))


print('\n              COMPARISON - harmonized values             ')
input('A vs B   - Press a key')
print(outA[outA.SITE=='UKBIOBANK'].mean(numeric_only = True))
print('  vs  ')
print(outB.mean(numeric_only = True))

print('\n              COMPARISON - harmonized values             ')
input('A vs C   - Press a key')
print(outA[outA.SITE=='BLSA_3T'].mean(numeric_only = True))
print('  vs  ')
print(outC.mean(numeric_only = True))

print('\n              COMPARISON - harmonized values             ')
input('A vs D   - Press a key')
print(outA[outA.SITE=='BLSA_3T'].mean(numeric_only = True))
print('  vs  ')
print(outD.mean(numeric_only = True))

print('\n              COMPARISON - harmonized values             ')
input('A vs F   - Press a key')
print(outA[outA.SITE=='BLSA_3T'].mean(numeric_only = True))
print('  vs  ')
print(outF[outF.SITE=='BLSATMP'].mean(numeric_only = True))

#######################################################################################
print('\nSave results')

## Save combined dataframes (init + harmonized data)
pd.concat([outA, dfROI_A], axis=1).to_csv('../Out/test1_df_out_A.csv', index = False)
pd.concat([outB, dfROI_B], axis=1).to_csv('../Out/test1_df_out_B.csv', index = False)
pd.concat([outC, dfROI_C], axis=1).to_csv('../Out/test1_df_out_C.csv', index = False)
pd.concat([outD, dfROI_D], axis=1).to_csv('../Out/test1_df_out_D.csv', index = False)
pd.concat([outE, dfROI_E], axis=1).to_csv('../Out/test1_df_out_E.csv', index = False)
pd.concat([outF, dfROI_F], axis=1).to_csv('../Out/test1_df_out_F.csv', index = False)



