# Imports
import os
from together import Together
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, Trainer, TrainingArguments, pipeline
import torch

CLIENT = Together(api_key="b924e54ec4b88f414bab35970ec734d82010c6aaf7258b3e3344ff5d374b7717")

CLASSIFIER = {"model": AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased"), 
              "tokenizer": AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")}

ENGINEER = "meta-llama/Llama-3-8b-chat-hf"

HIVE_MODELS = ["google/gemma-7b", "meta-llama/Llama-3-8b-hf", "togethercomputer/GPT-JT-Moderation-6B", "mistralai/Mistral-7B-v0.1", "codellama/CodeLlama-13b-Python-hf", "WizardLM/WizardCoder-Python-34B-V1.0"]

def classify_with_organizer(prompt):
    classifier_pipeline = pipeline(task="zero-shot-classification", model=CLASSIFIER["model"], tokenizer=CLASSIFIER["tokenizer"])
    output = classifier_pipeline(f"Classify this prompt to be passed into the best available large language model. \nPrompt: {prompt}", candidate_labels=list(HIVE_MODELS))
    labels = sorted([(output["labels"][i], output["scores"][i]) for i in range(len(output["labels"]))], key=lambda x: x[1], reverse=True)
    return labels # returns best to worst model in format (model_name, score)

def improve_prompt(prompt):
    response = CLIENT.chat.completions.create(
    model=ENGINEER,
    messages=[{"role": "system", "content": f"Rephrase any prompt you are given with prompt engineering. \
               Prompts should be improved to minimize the chance of response hallucination, maximize clarity, \
               and ensure depth of details. Just rephrase the prompt, no need to explain or say anything else. Here is the prompt: {prompt}"}],
    )
    return response.choices[0].message.content

def model_response(model_choice, prompt):
    response = CLIENT.completions.create(
    model=model_choice,
    prompt=prompt,
    max_tokens=512
    )
    return response.choices[0].text

def control_loop():
    print("Welcome to Hivemind! Please enter your prompt. If you want to exit, enter \'EXIT\'")
    while(True):
        inp = input("")
        if inp == "EXIT":
            break
        rephrased = improve_prompt(inp)
        model_choice = classify_with_organizer(rephrased)[0][0]
        print(model_response(model_choice, rephrased))

control_loop()