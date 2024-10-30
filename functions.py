import os
import mne

def data_load(dir, subjects, picks, avoid_overt=True) -> dict:
    '''This function takes in a directory, the desired subjects, the desired channels, and a boolean of whether or not to avoid overt trials
    i.e., those coded with 3 digits.
    It returns a dictionary with the data of the desired subjects and trials, indexed first by subject, and then by syllable.'''
    
    #initialize the dictionary
    data_dict = {}

    #loop through the subjects
    for subject in subjects:
        if subject not in data_dict:
            data_dict[subject] = {}
    
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.startswith(subject):
                    epo_name = file[10:-8]

                    if (avoid_overt == True) and (sum(c.isdigit() for c in epo_name) >= 3): #avoid the ones with 3 digits in them because those are the out-loud trials I believe - 
                                                                                            #will have to double check this but good to know if it works anyways
                        continue
            
                    file = os.path.join(dir, file)

                    data_dict[subject][epo_name] = mne.read_epochs(file, preload=True).pick(picks=picks).get_data()


    return data_dict
