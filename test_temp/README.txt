This is a temporary test folder for running nh_learn and nh_apply functions, and verifying that they produce the same output. Specifically to compare:
 - CASE 1: a batch is used as a training batch using nh_learn
 vs
 - CASE 2: the same site is used as an in-sample site using nh_apply
 vs
 - CASE 3: the same site is used as an out-of-sample site using nh_apply (same data, but batch name modified)
 
The test script displays all steps of this verification and important output variables.
The test notebook shows a simple plot of age trends for the harmonized data
