import json

with open('training/first_training_set/training_data.txt', 'r') as input_file:
    input_lines = input_file.readlines()
with open('training/first_training_set/training_labels.txt', 'r') as output_file:
    output_lines = output_file.readlines()

combined_data = []
for input_line, output_line in zip(input_lines, output_lines):
    combined_data.append({
        "text": input_line.strip(),
        "label": output_line.strip()
    })

with open('training/first_training_set/combined_training.json', 'w') as json_file:
    json.dump(combined_data, json_file, indent = 2)