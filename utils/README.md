# Utilities

This folder contains files that create class objects that load data from the `data` directory.

## Load Civil Comments

This file creates the CivilCommentsLoader class that loads the CivilCommentsWILDS dataset.  
The loader classifies a toxicity score of 0.5 or above as toxic and those below as non-toxic.
We return the sample id, text, and toxicity classification.

## Load Models

This file creates the ModelLoader class that loads models as Transformers Pipelines from checkpoint files.
These pipelines an be used as input to the SHAP library or to predict toxicity of text.

## Load SHAP Values 

This file creates the SHAPLoader class that loads SHAP values from a values.txt files and data.txt files that are the output of the SHAP_Generator notebook.
The data is returned as a pandas dataframe with the sample ID, token, toxic SHAP score and non-toxic SHAP score. 

## Load Words

This files creates the WordLoader class that loads words relating to demographics from the files in the `data` folder. 
