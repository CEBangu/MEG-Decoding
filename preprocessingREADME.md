# Preprocessing steps for Time-Frequency analysis

Let's follow best practices as laid out by MNE. 

Some notes about the raw data.
1. MEG 173 and MEG 059 are dead channels in all of the raw recordings. They are marked as bads.
2. The original sampling rate of the data is 2034.5100996195154; a lowpass filter at 1017.2550498097577 was applied when it was originally saved. 

## Filtering and Resampling

In this case, we are concerned with frequencies in the biological range. Namely, up to 150Hz - anything above this is considered noise. 

So first 
