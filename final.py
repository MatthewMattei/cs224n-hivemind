from together import Together
from keys import TOGETHER_API_KEY, CLASSIFIER_MODEL, STEM_MODEL, HUMANITIES_MODEL, SOCIAL_SCIENCES_MODEL, NICE_MODEL

CLIENT = Together(api_key=TOGETHER_API_KEY)

CLASSIFIER = CLASSIFIER_MODEL

HIVE_MODELS = {
    "STEM": STEM_MODEL,
    "humanities": HUMANITIES_MODEL,
    "social_sciences": SOCIAL_SCIENCES_MODEL,
    "nice": NICE_MODEL
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
