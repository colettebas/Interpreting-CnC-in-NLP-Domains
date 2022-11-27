 # final-project
UMass CS690F Final Project Fall 2022 - Interpreting Correct-n-Contrast in NLP Domains

### Project Description:
Spurious correlations can cause models to learn undesired relations in the data while training.  Often, these correlations negatively impact the way that models perform on minority groups in the data. Zhang et al. propose a novel method for removing spurious correlations in Correct-N-Contrast: A Contrastive Approach for Improving Robustness to Spurious Correlations which uses contrastive training without the need to identify the minority group. While automatically detecting minority samples is convenient for computer scientists, there is no way to tell if the model is correctly identifying the minority samples and which groups the model is helping. This project explores how the Correct-N-Contrast model performs in the NLP domain and attempts to predict which groups the contrastive model identified during training. To accomplish this, we compare SHAP values for the Correct-N-Contrast models to SHAP values from a standard ERM model using the Civil Comments-WILDS dataset. 

### Setting up the project:
This project currently consists of a demo that loads the NLP models trained by the Correct-n-Contrast authors.  The demo then uses the SHAP library to evaluate how these models make predictions on the Civil Comments-WILDS dataset. 

The next step for this project will be to select samples that will be to compare how the Correct-N-Contrast model and ERM model perform on samples containing minority group data.  This will help us identify which spurious correlations the Correct-n-Contrast removed through contrastive training. We then hope that we can use this information to identify the minority groups impacted by the Correct-n-Contrast training method. 


### Contributors
* Colette Basiliere
* Connor Brown
* Siddhant Shingi
* Mehek Tulsyan


