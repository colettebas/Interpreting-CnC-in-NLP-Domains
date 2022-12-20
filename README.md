# Interpreting Correct-n-Contrast in NLP Domains
UMass CS690F Final Project Fall 2022 - Interpreting Correct-n-Contrast in NLP Domains

### Project Description:
Spurious correlations can cause models to learn undesired relations in the data while training.  Often, these correlations negatively impact the way that models perform on minority groups in the data. Zhang et al. propose a novel method for removing spurious correlations in Correct-N-Contrast: A Contrastive Approach for Improving Robustness to Spurious Correlations which uses contrastive training without the need to identify the minority group. While automatically detecting minority samples is convenient for computer scientists, there is no way to tell if the model is correctly identifying the minority samples and which groups the model is helping. This project explores how the Correct-N-Contrast model performs in the NLP domain and attempts to predict which groups the contrastive model identified during training. To accomplish this, we compare SHAP values for the Correct-N-Contrast models to SHAP values from a standard ERM model using the Civil Comments-WILDS dataset. 

### Setting up the project:
This project currently consists of various methods that allow you to test NLP models by evaluating the SHAP values. 
We currently focus on comparing the ERM and Correct-n-Contrast models published by the authors of Corrent-n-Contrast. This codebase is designed
to allow for future studies to be conducting with different model or new samples. 
All of the experiments necessary to perform the analysis are located in the `./experiments` directory. 
To run these experiments on new models, users can simply add their own model checkpoint files that can be loaded as a Transformers pipeline.


After cloning the repo, make sure that you have installed all of the dependencies using `pip install -r requirements.txt` in the main directory.
The data to recreate these experiment can be downloaded at https://drive.google.com/drive/folders/1rcbefukUa0dd3XJNtf_73dTV0uUR_cdv?usp=sharing.
Store all the data files in the `./data` directory. The dataset is a subset of CivilCommentsWILDS. The models provded are those prodcued by the authors of
Correct-n-Contrast. New models and data can also be added to this directory.

The `./utils/` directory contains functions load the model pipelines and dataset that are used throughout the repository. 

The `SHAP_Generation.ipynb` file contains the code needed to save the SHAP values for the randomly
sampled samples to files for future use.  This process takes abut 16 hours so we provide the files produced
by this method in the Google Drive. The `./utils/` directory contains a function for loading the SHAP values from these files.

The `./experiments/` directory contains all of the files needed to produce the figures used in our paper. 
To simply user experience, we created a notebook that will automatically run these experiments. 
Once the data is placed in the correct directory, open the `Intreting_CnC_Visualizations.ipynb` notebook and Run All Cells to view the figures. 
The experiments in this notebook are labeled with the section they appear in our paper.
If you would like to update the experiments with new models or different samples, go into the corresponding experiment file 
and update the file paths or sample IDs.  


### Contributors
* Colette Basiliere
* Connor Brown
* Siddhant Shingi
* Mehek Tulsyan


