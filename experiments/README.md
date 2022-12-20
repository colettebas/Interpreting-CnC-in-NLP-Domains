# Experiments

This folder contains all of the files required to produce the files published in our paper.
These files all start by declaring variables that are the input to the various Loader classes in the `utils` folder. To customize 
these experiments, simply update these variables with the path to model or data files or update the sample IDs at the top of the main
method. 

## Plot Demographic Violin
This file creates violin charts of the SHAP values for CnC and ERM using words selected demographics. 
Plots are created for raw and normalized data. 

## Plot Difference in SHAP Vals by Sample
This file takes a sample ID and plots the difference in each SHAP value for each token in that sample.

## Plot Prediction
This file creates a graphs showing all sample prediction for CnC and ERM. This is useful for 
determining if one model has higher predictions than another. 

## Plot SHAP Comparison by Samples
This file creates a graph showing the SHAP values for each token in a sample side by side. This is useful
for determining how a model is altering the influence of SHAP values for a prediction.
Plots are created for raw and normalized data.

## Verify SHAP Sum
This file allows you to validate that the sum of the SHAP values for each sample plus the average
SHAP value for all samples equals the prediction value.  
