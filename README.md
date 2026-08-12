# SensitivityAnalysis_ecAraCore
Github repository accompanying the study of enzyme sensitivity coefficients in temperature adjusted ecAraCore (Lindner et al.). 

All scripts/data to reproduce results of the study can be found in this repository.

## /data
* Contains all of the necessary data sets / lookup tables used by other scripts.

## /models
* Contains temperature-adjusted ecAraCore models for the temperature range 10°C-40°C

## /results
* Output directory for datasets/tables (/tables) and figures (/figures)

## /scripts
* Contains all scripts to generate and preprocess data sets (/dataGen) and to generate results (/resultGen)

## /source
Contains python classes:
### GEM()
* python representation of enzyme constrained GEMs in matlab struct format.
* contains several QOL functions to explore model sctructure

### OptimizationProblem(GEM)
* Summarizes basic flux-based optimizations implemented with the gurobi solver (FBA, FVA, pFBA).

### SensitvityAnalyis(GEM)
* Class to compute all sensitivity coefficients (flux capacity, enzyme capacity, proteome capacity, enzyme) for general GECKO-style ecGEMs/pcGEMS.

### createTGEMs.m (MATLAB)
* script to create the temperature adjusted models of ecAraCore, using the baseline TGEM model (Wendering et al. 2025)
* This script uses functions from the git repository of https://doi.org/10.1111/nph.20420.

### __init__
* Collection of constants for file paths to all files used by two or more scripts.
