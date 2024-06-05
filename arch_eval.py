# Imports
from together import Together
from datasets import Dataset
import pandas as pd
import numpy as np
import google.generativeai as genai
from keys import TOGETHER_API_KEY

CLASSIFIER = ""

CLIENT = Together(api_key=TOGETHER_API_KEY)

MODEL_OPTIONS = {
    "STEM": "",
    "humanities": "",
    "other": "",
    "social sciences": ""
}

def classify(prompt):
    response = CLIENT.completions.create(
    model=CLASSIFIER,
    prompt=prompt,
    max_tokens=512
    )
    return response.choices[0].text.split(" ")[-1]

def respond(prompt, model_choice):
    response = CLIENT.completions.create(
    model=model_choice,
    prompt=prompt,
    max_tokens=512
    )
    return int(response.choices[0].text.split(" ")[-1])

def eval_architecture(data_path):
    