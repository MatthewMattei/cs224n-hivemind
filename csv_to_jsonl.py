import csv
import json

def csv_to_jsonl(input_csv_file, output_jsonl_file):
    # Open the CSV file and read its contents
    with open(input_csv_file, 'r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Open the JSONL file for writing
        with open(output_jsonl_file, 'w') as jsonl_file:
            for row in csv_reader:
                # Convert each row (a dictionary) to a JSON object
                json_object = json.dumps(row)
                
                # Write the JSON object to the JSONL file
                jsonl_file.write(json_object + '\n')

# Usage example:
input_csv_file = 'input.csv'
output_jsonl_file = 'output.jsonl'
csv_to_jsonl(input_csv_file, output_jsonl_file)
