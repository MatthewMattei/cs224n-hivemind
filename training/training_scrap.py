'''
File: training_scrap.py
---------------------------------
Contains functions to produce training dataset information.
'''
import pandas as pd
import random
import re

output_file = "training/training_data.txt"
total_input_lines = 100

training_phrases = []

# Cleans string of invalid characters
def clean_string(string_input: str):
    string_input = string_input.strip()
    string_input = string_input.replace('\n', '').strip()
    string_input = string_input.replace(':', '').strip()
    re.sub(r'[^\x00-\x7f]',r'', string_input)
    return string_input


# Empties the training_data.txt before adding to it
with open(output_file, 'w') as file:
    pass

# Adds training_info.csv file ##############################################

training_file = "training/training_data/training_info.csv"

data_frame = pd.read_csv(training_file)
desired_training_prompt = data_frame["instruction"]
desired_training_inputs = data_frame["input"]

for i in range(len(desired_training_inputs)):
    training_prompt = desired_training_prompt[i]
    training_input = desired_training_inputs[i]

    training_prompt = clean_string(training_prompt)
    if (not pd.isna(training_input) and training_input != "Not applicable"):
        training_input = clean_string(training_input)
        new_input = training_prompt + " This is what the type of input would look like: " + training_input
    
    else:
        new_input = training_prompt
    
    training_phrases.append(new_input)

# Adds prompt_tuning_train.csv file ##############################################

training_file = "training/training_data/prompt_tuning_train.csv"

data_frame = pd.read_csv(training_file)
desired_training_prompt = data_frame["improved_prompts"]

# Loops through all the desired training inputs, clean inputs
for i in range(len(desired_training_prompt)):
    training_prompt = desired_training_prompt[i]

    if (not pd.isna(training_input)):
        training_prompt = clean_string(training_prompt)
        training_phrases.append(training_prompt)

# Randomly shuffles the training_phrases (for variety)
random.shuffle(training_phrases)

with open(output_file, 'w') as file:
    for i in range(total_input_lines):
        if (i == total_input_lines - 1):
            file.write(training_phrases[i])
        else:
            file.write(training_phrases[i] + '\n')