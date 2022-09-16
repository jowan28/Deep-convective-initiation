# Code for Identifying, and predicting Deep convective initiation

The code was run on the Jasmin servers.
The shell code references the python environment jasenv.
jasenv requires the tobac-flow module from the cloned repository https://github.com/w-k-jones/tobac-flow accesed 16/09/2022

The code is split into:
-Data extraction and formatting
-model and optimization
-cross validation and plotting

Everything except plotting is done via Python files with an accompanying shell file that allows the parallelisation of calculations.
Plotting is done in Jupyter notebooks
