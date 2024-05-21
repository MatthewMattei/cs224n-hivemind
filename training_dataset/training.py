import pandas as pd
import random

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
                # Clears out any extra line or colon characters
                training_input = training_input.strip()
                training_input = training_input.replace('\n', '').strip()
                training_input = training_input.replace(':', '').strip()

                # Adds cleaned prompt to file
                file.write(training_input + '\n')

# Codegemma file extraction
codegemma_files = ['training_dataset/codegemma_datasets/code_dataset.csv']
codegemma_columns = ['Problem']

codegemma_training = 'training_dataset/codegemma_training.txt'

add_to_files(codegemma_files, codegemma_columns, codegemma_training)

# Llama2 file extractions
llama2_files = ["training_dataset/llama2_datasets/first_dataset.csv", "training_dataset/llama2_datasets/second_dataset.csv", 
                "training_dataset/llama2_datasets/third_dataset.csv"]
llama2_columns = ['content', 'content', 'content']

llama2_training = 'training_dataset/llama2_training.txt'

add_to_files(llama2_files, llama2_columns, llama2_training)

# Llama3 file extractions
llama3_files = ['training_dataset/llama3_datasets/quotes.csv']
llama3_columns = ['quote']

llama3_training = 'training_dataset/llama3_training.txt'

add_to_files(llama3_files, llama3_columns, llama3_training)