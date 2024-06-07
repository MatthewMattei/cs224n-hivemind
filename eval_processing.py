# Imports
import json

DATA_PATH = "CLASSIFY3_EVALS.jsonl"
check = "overall_eval_2_selection_unformatted.jsonl"

results = []
f = open(DATA_PATH, 'r')
lines = f.readlines()
for i, line in enumerate(lines):
     results.append(json.loads(line)[str(i)])

answers = []
f = open(check, 'r')
lines = f.readlines()
for i, line in enumerate(lines):
    answers.append((json.loads(line)["answer"], json.loads(line)["subject"]))

with open("processed_" + DATA_PATH, 'w') as jsonl_file:
    correct_total = 0
    correct_stem = 0
    correct_humanities = 0
    correct_social_sciences = 0
    for i in range(len(results)):
        outcome = "correct" if str(results[i]) == answers[i][0] or str(results[i]) == answers[i][1] else "incorrect"
        if results[i] == "social_sciences" and "social sciences" == answers[i][1]:
            outcome = "correct"
        if outcome == "correct":
            correct_total += 1
            if answers[i][1] == "STEM":
                correct_stem += 1
            elif answers[i][1] == "humanities":
                correct_humanities += 1
            elif answers[i][1] == "social sciences":
                correct_social_sciences += 1
        json_line = json.dumps({"Question": str(i), "Guessed_Answer": str(results[i]), "Correct_Answer": answers[i], "Outcome": outcome})
        jsonl_file.write(json_line + '\n')
    json_line = json.dumps({"Total_Correct": str(correct_total),
                            "STEM_Correct": str(correct_stem),
                            "HUMANITIES_Correct": str(correct_humanities),
                            "SOCIAL_SCIENCES_Correct": str(correct_social_sciences)})
    jsonl_file.write(json_line + '\n')