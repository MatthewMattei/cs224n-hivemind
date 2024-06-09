#Calculation of what the scores would be if a consensus between all four models is picked

import json
import random

paths = ["processed_CLASSIFY2_EVALS.jsonl", "processed_STEM_EVALS.jsonl", "processed_HUMANITIES_EVALS.jsonl", "processed_SOCIAL_SCIENCES_EVALS.jsonl", "processed_OTHER_EVALS.jsonl"]

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

new_file_path = "random_choice_total_consensus_eval.jsonl"

def determine_consensus(choice1, answer1, choice2, answer2, choice3, answer3, choice4, answer4):
    choices = [choice1, choice2, choice3, choice4]
    answers = [answer1, answer2, answer3, answer4]
    final_choices = []
    if (answer1 != answer2 != answer3 != answer4):
        return random.choice(choices)
    if (answer1 == answer2 == answer3 == answer4):
        return choice1
    else:
        answer_count = [0, 0, 0, 0]
        for answer in answers:
            answer_count[int(answer)] += 1
        consensus = answer_count.index(max(answer_count))
        for i in range(4):
            if answers[i] == str(consensus):
                final_choices.append(choices[i])
                return random.choice(final_choices)


for i in range(5):
    mega_lines.append(open(paths[i], 'r').readlines())

for i in range(1000):
    model_pickings = ["STEM", "humanities", "social sciences", "other"]
    selection1 = json.loads(mega_lines[1][i])["Guessed_Answer"]
    selection2 = json.loads(mega_lines[2][i])["Guessed_Answer"]
    selection3 = json.loads(mega_lines[3][i])["Guessed_Answer"]
    selection4 = json.loads(mega_lines[4][i])["Guessed_Answer"]
    choice = determine_consensus("STEM", selection1, "humanities", selection2, "social sciences", selection3, "other", selection4)
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

