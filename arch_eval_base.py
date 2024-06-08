# Imports
from together import Together
from keys import TOGETHER_API_KEY
import json
import requests

MODEL = "meta-llama/Llama-3-8b-hf"

CLIENT = Together(api_key=TOGETHER_API_KEY)

DATA_PATH = "overall_eval_selection.jsonl"

def get_model_response(prompt):
    prompt = "<human>: I am about to ask you a multiple choice question. Respond with the number associated with your answer choice (0, 1, 2, or 3) and nothing else. " + prompt[8:]
    try:
        response = CLIENT.completions.create(
        model=MODEL,
        prompt=prompt,
        max_tokens=1
        )
        return response.choices[0].text.split(" ")[-1]
    except requests.RequestException as e:
        print(f"Error making requests to {prompt, MODEL}: {e}")
        return ""

results = []

f = open(DATA_PATH, 'r')
lines = f.readlines()
for i, line in enumerate(lines):
     print("Finished: " + str(i))
     if i < 10:
        print(results)
     results.append(get_model_response(json.loads(line)["text"]))
    
jsonl_file_path = f'{"BASE"}_EVALS.json'
with open(jsonl_file_path, 'w') as jsonl_file:
        for key, value in enumerate(results):
            json_line = json.dumps({key: value})
            jsonl_file.write(json_line + '\n')
    
print(f"Results saved to {jsonl_file_path}")