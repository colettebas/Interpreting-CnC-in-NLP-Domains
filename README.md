 # final-project
UMass CS690F Final Project Fall 2022 - Interpreting Correct-n-Contrast in NLP Domains

### Project Description:
Spurious correlations can cause models to learn undesired relations in the data while training.  Often, these correlations negatively impact the way that models perform on minority groups in the data. Zhang et al. propose a novel method for removing spurious correlations in Correct-N-Contrast: A Contrastive Approach for Improving Robustness to Spurious Correlations which uses contrastive training without the need to identify the minority group. While automatically detecting minority samples is convenient for computer scientists, there is no way to tell if the model is correctly identifying the minority samples and which groups the model is helping. This project explores how the Correct-N-Contrast model performs in the NLP domain and attempts to predict which groups the contrastive model identified during training. To accomplish this, we compare SHAP values for the Correct-N-Contrast models to SHAP values from a standard ERM model using the Civil Comments-WILDS dataset. 

### Setting up the project:
This project currently consists of various methods that allow you to test NLP models by evaluating the SHAP values. 
We currently focus on comparing the ERM and Correct-n-Contrast models published by the authors of Corrent-n-Contrast.
After cloning the repo, make sure that you have installed all of the dependencies using `pip install -r requirements.txt` in the main directory.


There is a demo that computes the SHAP values for the NLP models trained by the Correct-n-Contrast authors.  
The demo creates pipelines for each model along with SHAP explainers to compute SHAP values 
for specific samples in the Civil Comments-WILDS dataset. 

The remaining files complete various tasks to compute metrics that are useful in evaluating models.  
These files use the utilites in utils to load the data, load the models, and load the SHAP values.  The data 
to recreate these experiment can be downloaded at https://drive.google.com/drive/folders/1rcbefukUa0dd3XJNtf_73dTV0uUR_cdv?usp=sharing
The files downloaded from this folder should be placed in a /data folder in the root directory.

###Verify SHAP Sum
This file allows you to validate that the sum of the SHAP values for each sample plus the average
SHAP value for all samples equals the prediction value.  To use this file, update the model filename, 

### Contributors
* Colette Basiliere
* Connor Brown
* Siddhant Shingi
* Mehek Tulsyan


