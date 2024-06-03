"""
TODO:
1. make it possible to load models from huggingface
2. make it possible to generate content from models
3. create class to organize models into different categories
4. create program flow that passes user input to organizer, does work, then passes output to a ranked model
5. create program flow that allows for evaluation of organizer model
6. create finetuning control flow
"""
# Imports
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, Trainer, TrainingArguments, pipeline
import torch

# huggingface name of organizer model - required for any control flow, set this to pathname for local finetuned use
ORGANIZER_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
# huggingface name of evaluator model - required only for finetuning
EVALUATOR_MODEL_NAME = ""
# huggingface names of hive models - required for any control flow
HIVE_MODEL_NAMES = {"google-t5/t5-base": "translation", "codellama/CodeLlama-7b-Python-hf": "text-generation"}
# text descriptions of hive models - useful if organizer is not finetuned
HIVE_MODEL_DESCS = {
        "google-t5/t5-base": "With T5, we propose reframing all NLP tasks into a unified text-to-text-format where the input and output are always text strings, in contrast to BERT-style models that can only output either a class label or a span of the input. Our text-to-text framework allows us to use the same model, loss function, and hyperparameters on any NLP task.",
        "codellama/CodeLlama-7b-Python-hf": "Code Llama is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 34 billion parameters. This is the repository for the 7B Python specialist version in the Hugging Face Transformers format. This model is designed for general code synthesis and understanding. Links to other models can be found in the index at the bottom.",
}

def classify_with_organizer(organizer, hive, prompt):
    classifier_pipeline = pipeline(task="zero-shot-classification", model=organizer["classify_model"], tokenizer=organizer["tokenizer"], device_map="auto")
    output = classifier_pipeline(f"Classify this prompt to be passed into the best available large language model. \nPrompt: {prompt}", candidate_labels=list(hive.keys()))
    labels = sorted([(output["labels"][i], output["scores"][i]) for i in range(len(output["labels"]))], key=lambda x: x[1], reverse=True)
    return labels # returns best to worst model in format (model_name, score)

def rephrase_with_organizer(organizer, prompt):
    rephrase_pipeline = pipeline(task="text-generation", model=organizer["rephrase_model"], tokenizer=organizer["tokenizer"], device_map="auto")
    combined_text = f"The following is a prompt to be passed into a large language model. Use prompt \
    engineering tricks, word choice changes, and clarification to change the prompt into something that \
        will be more effectively answered (respone should have no hallucination, have more details, and be more accurate).\n \
    Original prompt: {prompt}"
    output = rephrase_pipeline(combined_text)
    return output["sequence"] # returns rephrased prompt text

def respond_with_hive(hive, model_choice, prompt):
    response_pipeline = pipeline(model=hive[model_choice]["model"], tokenizer=hive[model_choice]["tokenizer"], device_map="auto")
    output = response_pipeline(prompt)
    return output["sequence"] # returns response text

if __name__ == '__main__':

    try:
        organizer = {
            "classify_model": AutoModelForSequenceClassification.from_pretrained(ORGANIZER_MODEL_NAME, device_map="auto"),
            "tokenizer": AutoTokenizer.from_pretrained(ORGANIZER_MODEL_NAME, device_map="auto"),
            "rephrase_model": AutoModelForCausalLM.from_pretrained(ORGANIZER_MODEL_NAME, device_map="auto")
        }
    except:
        print("Organizer model failed to load. Model name entered: " + ORGANIZER_MODEL_NAME)
        exit()

    evaluator = {
        "model": AutoModelForSequenceClassification.from_pretrained(EVALUATOR_MODEL_NAME, device_map="auto"),
        "tokenizer": AutoTokenizer.from_pretrained(EVALUATOR_MODEL_NAME, device_mao="auto")
    } if EVALUATOR_MODEL_NAME != "" else {"model": None, "tokenizer": None}

    hive = {}
    try:
        curr_model = ""
        for name in HIVE_MODEL_NAMES:
            curr_model = name
            HIVE_MODEL_DESCS[name] # will crash out if there isn't a description for a model name
            hive[name] = {
                "model": AutoModel.from_pretrained(name, device_map="auto"),
                "tokenizer": AutoTokenizer.from_pretrained(name, device_map="auto")
            }
    except:
        print("Error loading hive models. Last model attempted: " + curr_model)
        exit()

    #sample_rephrase = rephrase_with_organizer(organizer=organizer, prompt="Translate the following into Chinese: Throw that dumpster trash into the garbage!")
    model_choice = classify_with_organizer(organizer=organizer, hive=hive, prompt="Translate the following into Chinese: Throw that dumpster trash into the garbage!")
    print(model_choice, model_choice[0][0])
    print(respond_with_hive(hive, model_choice[0][0], "Translate the following into Chinese: Throw that dumpster trash into the garbage!"))
    print(model_choice)