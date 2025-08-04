import os
import pandas as pd 
import mne


def main():

    path = "/pasteur/appa/scratch/cbangu/scalograms/overt/16x16/data"

    speech_vowel_dict = {}

    for f in os.listdir(path):
        parsed_f = f.split("_")
        vowel = parsed_f[3]
        speech_type = len(parsed_f[4])

        if speech_type not in speech_vowel_dict:
            speech_vowel_dict[speech_type] = {}
            speech_vowel_dict[speech_type][vowel] = 1
        
        elif vowel not in speech_vowel_dict[speech_type]:
            speech_vowel_dict[speech_type][vowel] = 1
        
        else:
            speech_vowel_dict[speech_type][vowel] += 1

    print("TOTALS")
    print(speech_vowel_dict)

    speech_vowel_dict_sub_01 = {}

    for f in os.listdir(path):
        parsed_f = f.split("_")
        if parsed_f[1]=='01':
            vowel = parsed_f[3]
            speech_type = len(parsed_f[4])

            if speech_type not in speech_vowel_dict_sub_01:
                speech_vowel_dict_sub_01[speech_type] = {}
                speech_vowel_dict_sub_01[speech_type][vowel] = 1
            
            elif vowel not in speech_vowel_dict_sub_01[speech_type]:
                speech_vowel_dict_sub_01[speech_type][vowel] = 1
            
            else:
                speech_vowel_dict_sub_01[speech_type][vowel] += 1
        else:
            continue

    print('\n')
    print("Subject 1")
    print(speech_vowel_dict_sub_01)
if __name__ == "__main__":
    main()
