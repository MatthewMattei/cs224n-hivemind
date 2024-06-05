import json
import random

def merge_and_shuffle_jsonl_files(input_files, output_file):
    all_lines = []

    # Read lines from each input file
    for input_file in input_files:
        with open(input_file, 'r') as infile:
            lines = infile.readlines()
            all_lines.extend(lines)

    # Shuffle the lines randomly
    random.shuffle(all_lines)

    # Write shuffled lines to the output file
    with open(output_file, 'w') as outfile:
        for line in all_lines:
            # Ensure the line is a valid JSON before writing
            try:
                json_object = json.loads(line)
                outfile.write(json.dumps(json_object) + '\n')
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line}")
                continue

input_files = [
    '/Users/matthsu/Documents/GitHub/cs224n/humanities_eval_selection.jsonl',
    '/Users/matthsu/Documents/GitHub/cs224n/other_eval_selection.jsonl',
    '/Users/matthsu/Documents/GitHub/cs224n/social_science_eval_selection.jsonl',
    '/Users/matthsu/Documents/GitHub/cs224n/stem_eval_selection.jsonl'
]
output_file = '/Users/matthsu/Documents/GitHub/cs224n/overall_eval_selection_unformatted.jsonl'
    
merge_and_shuffle_jsonl_files(input_files, output_file)
print(f"Merged and shuffled files into {output_file}")
