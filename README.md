# Decoding Speech from Magnetoencephalography Data

## Summary
This repository contains code relevant to my Master’s thesis, conducted at the Neurospeech lab at the Pasteur Institute de l’Audition, and supervised by Dr. Anne-Lise Giraud, and PhD student Soufiane Jhilal.

This thesis investigated the applicability of Computer Vision models to accurately classify covertly spoken vowels from the scalogram transform of their MEG representation, with an eye towards applications in Brain Computer Interfaces. It explored decoding both sensor space representations – i.e., the scalograms obtained from each MEG sensor, as well as source space representations – i.e., the scalograms from data obtained via source-reconstruction (LCMV-beamforming). 

While the Broca’s area representation contains hints of discernibility, overall better than chance accuracy was not achieved. This result highlights the inherent difficulties with extra-cranial decoding in fine-target settings such as covertly spoken vowels. 

## Repo Structure
While the experiment cannot be recreated from this code alone (proprietary data and SLURM cluster required). However, the main scripts are all found here.  

See `datahandling` for data classes, `experiment` for experiment functions, `models` for model definitions, `plotting` for the plotting class, `preprocessing` for the preprocessing helpers written by Soufiane, and `wavelets` for the wavelet helpers. 

