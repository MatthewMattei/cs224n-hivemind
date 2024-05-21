'''
File: training.py
---------------------------------
Contains functions to produce training dataset information. Currently, all of this data will be used for
evaluation.

** Possibly will be rename to include extracting training set information to test set and evaluation sets as well
** Training: 80%, Evaluation: 20%
'''
import pandas as pd
import random
import re

'''
Method: add_to_files
---------------------------------
Converts CSV file to txt file to combine all training data information.
This function will split the CSV files into it's desired models txt file.
Parameters:
- files: list of relative addresses for each of the training data files
- columns: list of columns in the training data files that corresponds to what we want to extract
- output_file: where these combined inputs should go to (the training data txt file)
'''
def add_to_files(files: list, columns: list, output_file: str):
    # Empties the training_files before adding to it
    with open(output_file, 'w') as file:
        pass

    # Loops through all the files in the list of CSV files
    for idx in range(len(files)):
        single_file = files[idx]
        column = columns[idx]
        data_frame = pd.read_csv(single_file)
        desired_training_inputs = data_frame[column]

        # Writes into the output files
        with open(output_file, 'a') as file:
            # Loops through all the desired training inputs from the single file
            for training_input in desired_training_inputs:
                # Clears out any extraneous characters
                training_input = training_input.strip()
                training_input = training_input.replace('\n', '').strip()
                training_input = training_input.replace(':', '').strip()
                re.sub(r'[^\x00-\x7f]',r'', training_input)

                # Adds cleaned prompt to file
                file.write(training_input + '\n')

# Codegemma file extraction
codegemma_files = ["training_dataset/codegemma_datasets/code_dataset.csv", "training_dataset/codegemma_datasets/prompt_tuning_train.csv",
                    "training_dataset/codegemma_datasets/prompts-test.csv"]
codegemma_columns = ["Problem", "improved_prompts", "improved_prompts"]

codegemma_training = "training_dataset/codegemma_training.txt"

add_to_files(codegemma_files, codegemma_columns, codegemma_training)

# Llama2 file extractions
llama2_files = ["training_dataset/llama2_datasets/first_dataset.csv", "training_dataset/llama2_datasets/second_dataset.csv", 
                "training_dataset/llama2_datasets/third_dataset.csv", "training_dataset/llama2_datasets/dataset_filtered.csv"]
llama2_columns = ["content", "content", "content", "Text"]

llama2_training = "training_dataset/llama2_training.txt"

add_to_files(llama2_files, llama2_columns, llama2_training)

# Llama3 file extractions
llama3_files = ["training_dataset/llama3_datasets/quotes.csv", "training_dataset/llama3_datasets/dataset_filtered.csv",
                "training_dataset/llama3_datasets/cefr_leveled_texts.csv", "training_dataset/llama3_datasets/prompt_tuning.csv"]
llama3_columns = ["quote", "Text", "text", "improved_prompts"]

llama3_training = "training_dataset/llama3_training.txt"

add_to_files(llama3_files, llama3_columns, llama3_training)

# Randomization and adding training information and actual answers to the evaluation file
evaluation_data = "training_dataset/evaluation.txt"
evaluation_actual_answer = "training_dataset/evaluation_answers.txt"

codegemma_info = []
llama2_info = []
llama3_info = []

# Appends information from the dataset files to lists
with open("training_dataset/codegemma_training.txt", 'r') as file:
    for line in file:
        codegemma_info.append(line.strip())
with open("training_dataset/llama2_training.txt", 'r') as file:
    for line in file:
        llama2_info.append(line.strip())
with open("training_dataset/llama3_training.txt", 'r') as file:
    for line in file:
        llama3_info.append(line.strip())

# Cleans any previous information in the evaluation input data and evaluation actual answer file
with open(evaluation_data, 'w') as file:
    pass
with open(evaluation_actual_answer, 'w') as file:
    pass

evaluation_data = "training_dataset/evaluation.txt"
evaluation_actual_answer = "training_dataset/evaluation_answers.txt"

combined_info = [(info, 'codegemma') for info in codegemma_info] + \
                [(info, 'llama2') for info in llama2_info] + \
                [(info, 'llama3') for info in llama3_info]

# Randomly shuffles all the information from the lists with it's associated language
random.shuffle(combined_info)

# Splits the two files: one for all evaluation information dn one for the answers associated to the evaluation information
with open(evaluation_data, 'w') as eval_file, open(evaluation_actual_answer, 'w') as ans_file:
    for info, source in combined_info:
        eval_file.write(info + '\n')
        ans_file.write(source + '\n')