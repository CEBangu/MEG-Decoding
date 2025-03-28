### Can covertly spoken syllables be decoded from MEG data?

This repository aims to answer this question. It is also my Master's Thesis. 

Projected completion date: June 2025 

## Workflow:

1. Compute scalorgam coefficiens using coefficient_computation.py
2. Generate the scalograms using plotting.py
3. Split the data into Test and Train/Val. 
4. Train the models / find the best hyperparameters using the training loop: cnn_train.py, vit_train.py
5. Test the models using: cnn_test.py, vit_test.py (TBC)
6. Run the statistical tests on their performance with stats.py (TBC)

## Current Organization:

**NB!** some of the files will be broken because of path references/import errors because of running it from the server/changes in organization. I guess these will be fixed 1 by 1. 

Currently, it's a bit of a mess. But here is a bit of an overview of the main ones:

### Subdirectories:
datahandling contains class to handle data structures, the most common ones now are BcomMEG, which lets you handle the participant data in a clean way, and AlexNetDataHandler which is the dataclass for the AlexNet fine-tune. 

Data_Sample contains some participant data to do quick testing on

images contains some plots generated by the slice_tca tests

models contains model classes

notebooks contains three subdirectories of previous work: data_exploration, model_experiments and cluster_tests.

scalograms_test contains the generated scalograms used to test out the model training scripts locally

### Individual Files:
The coefficients for testing the plotting scripts are the files in coefficients.npy

The notebooks contain the ideation for the scripts that will eventually be made. 

