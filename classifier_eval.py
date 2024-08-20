import json

paths = ["processed_CLASSIFIER_EVALS.jsonl", "processed_STEM_EVALS.jsonl", "processed_HUMANITIES_EVALS.jsonl", "processed_SOCIAL_SCIENCES_EVALS.jsonl", "processed_OTHER_EVALS.jsonl"]

mega_lines = []

total_correct = 0
stem_total = 0
stem_correct = 0
humanities_total = 0
humanities_correct = 0
social_total = 0
social_correct = 0
other_total = 0
other_correct = 0
misclassified_correct = 0
misclassified_wrong = 0

new_file_path = "original_classifier_pipeline_eval.jsonl"

for i in range(5):
    mega_lines.append(open(paths[i], 'r').readlines())

for i in range(1000):
    choice = json.loads(mega_lines[0][i])["Guessed_Answer"]
    if choice == "STEM":
        stem_total += 1
        if json.loads(mega_lines[1][i])["Outcome"] == "correct":
            total_correct += 1
            stem_correct += 1
    if choice == "humanities":
        humanities_total += 1
        if json.loads(mega_lines[2][i])["Outcome"] == "correct":
            total_correct += 1
            humanities_correct += 1
    if choice == "social sciences":
        social_total += 1
        if json.loads(mega_lines[3][i])["Outcome"] == "correct":
            total_correct += 1
            social_correct += 1
    if choice == "other":
        other_total += 1
        if json.loads(mega_lines[4][i])["Outcome"] == "correct":
            total_correct += 1
            other_correct += 1

with open(new_file_path, 'w') as jsonl_file:
    json_line = json.dumps({"total_correct": str(total_correct), 
                            "stem_total": str(stem_total), 
                            "stem_correct": str(stem_correct), 
                            "humanities_total": str(humanities_total), 
                            "humanities_correct": str(humanities_correct), 
                            "social_total": str(social_total), 
                            "social_correct": str(social_correct), 
                            "other_total": str(other_total), 
                            "other_correct": str(other_correct),
                            })
    jsonl_file.write(json_line + '\n')

