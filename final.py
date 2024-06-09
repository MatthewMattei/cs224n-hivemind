# Imports
from together import Together
from keys import TOGETHER_API_KEY

CLIENT = Together(api_key=TOGETHER_API_KEY)

CLASSIFIER = "mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-06-21-14-24-9a675604"

HIVE_MODELS = {
    "STEM": "mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-05-00-23-38-c6149bf9",
    "humanities": "mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-04-23-25-37-12c540c5",
    "social_sciences": "mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-04-23-26-33-e15174ab",
    "nice": "mdmattei@stanford.edu/Meta-Llama-3-8B-2024-06-04-23-20-24-4c9b70b7"
}

def classify(prompt):
    try:
        response = CLIENT.completions.create(
        model=CLASSIFIER,
        prompt=prompt,
        max_tokens=1
        )
        return response.choices[0].text.split(" ")[-1]
    except:
        print(f"Error making requests to {prompt, CLASSIFIER}")
        return ""

def get_model_response(prompt, model_choice):
    try:
        response = CLIENT.completions.create(
        model=HIVE_MODELS[model_choice],
        prompt=prompt,
        max_tokens=1
        )
        return response.choices[0].text.split(" ")[-1]
    except:
        print(f"Error making requests to {prompt, CLASSIFIER}")
        return ""


print("Enter question below:\n")

while(True):
    prompt = input()
    if prompt == "EXIT":
        break
    model_choice = classify(prompt)
    resp = get_model_response(prompt, model_choice)
    print(f"Chosen Model: {model_choice} - Answer: {resp}")
