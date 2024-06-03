'''
File: training_script.py
---------------------------------
Runs information in training_data.txt through provided HIVE_MODELS through TOGETHER_API_KEY.
'''

import os
from together import Together
from together.error import APIError

TOGETHER_API_KEY = "redacted"
HIVE_MODELS = ['WizardLM/WizardCoder-Python-34B-V1.0', 'Phind/Phind-CodeLlama-34B-v2', 'codellama/CodeLlama-70b-Instruct-hf']
TRAINING_DATA = "training/training_data.txt"
MODEL_OUTPUT = "training/training_outputs.txt"

client = Together(api_key=TOGETHER_API_KEY)

# Opens both input and output files
with open(TRAINING_DATA, 'r') as input_file, open(MODEL_OUTPUT, 'w') as output_file:
    counter = 1

    # Goes through every line of the input file
    for line in input_file:
        # Writes what test we are on in terminal and output file
        print("#################################################################################### \n")
        print(str(counter) + '\n')
        print(line)

        output_file.write("#################################################################################### \n")
        output_file.write(str(counter) + '\n')
        output_file.write(line)

        # Goes through every model provided
        for model in HIVE_MODELS:
            print(model + '\n')
            output_file.write(model + '\n')

            # Sees if model will provide a valid response. Prints error if not
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": line}],
                )
                print(response.choices[0].message.content)
                print('\n')
                output_file.write(response.choices[0].message.content)
                output_file.write('\n')
            except APIError as e:
                print(f"APIError occurred: {e}")
                output_file.write(f"APIError occurred: {e}\n")
            except Exception as e:
                print(f"An error occurred: {e}")
                output_file.write(f"An error occurred: {e}\n")

        # Increment test number.
        counter += 1
            