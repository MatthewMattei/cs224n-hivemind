import json
import random

def select_random_lines(input_file, output_file, num_lines=333):
    # Read all lines from the input JSONL file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Randomly select 1000 lines (or fewer if the file has less than 1000 lines)
    selected_lines = random.sample(lines, min(num_lines, len(lines)))
    
    # Write the selected lines to the output JSONL file
    with open(output_file, 'w') as f:
        for line in selected_lines:
            f.write(line)

# Usage example:
input_file = '/Users/matthsu/Documents/GitHub/cs224n/humanities_eval_2.jsonl'
output_file = '/Users/matthsu/Documents/GitHub/cs224n/humanities_eval_2_selection.jsonl'
select_random_lines(input_file, output_file)
