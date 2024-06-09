# Imports
from together import Together
from transformers import BertTokenizer, BertForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import pandas as pd
import numpy as np
import evaluate
import google.generativeai as genai
from keys import TOGETHER_API_KEY, GEMINI_API_KEY

CLIENT = Together(api_key=TOGETHER_API_KEY)

CLASSIFIER = {"model": BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased"), 
              "tokenizer": BertTokenizer.from_pretrained("google-bert/bert-base-uncased")}

ENGINEER = genai.GenerativeModel('gemini-1.0-pro')

HIVE_MODELS = ["Phind/Phind-CodeLlama-34B-v2", "codellama/CodeLlama-70b-Instruct-hf", "WizardLM/WizardCoder-Python-34B-V1.0"]

def classify_with_organizer(prompt):
    classifier_pipeline = pipeline(task="zero-shot-classification", model=CLASSIFIER["model"], tokenizer=CLASSIFIER["tokenizer"])
    output = classifier_pipeline(f"Classify this prompt to be passed into the best available large language model. Clearly explain why you choose the model you've chosen. \nPrompt: {prompt}. \nModels: {HIVE_MODELS}", candidate_labels=list(HIVE_MODELS))
    labels = sorted([(output["labels"][i], output["scores"][i]) for i in range(len(output["labels"]))], key=lambda x: x[1], reverse=True)
    return labels # returns best to worst model in format (model_name, score)

def improve_prompt(prompt):
    response = ENGINEER.generate_content(f"Rephrase any prompt you are given with prompt engineering. \
               Prompts should be improved to minimize the chance of response hallucination, maximize clarity, \
               and ensure depth of details. Just rephrase the prompt, no need to explain or say anything else. Here is the prompt: {prompt}")

    return response.text

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

# Dataset formatting instructions - https://huggingface.co/docs/transformers/en/tasks/sequence_classification#train
def finetune_classifier(dataset_path):
    def load_dataset_from_csv(file_path):
        df = pd.read_json(file_path)
        return Dataset.from_pandas(df)
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    def tokenize_dataset(dataset, tokenizer):
        def preprocess_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True)
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        return tokenized_dataset
    
    dataset = load_dataset_from_csv(dataset_path)
    dataset = dataset.train_test_split(test_size=0.2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")
    id2label = {0: "Phind/Phind-CodeLlama-34B-v2", 1: "codellama/CodeLlama-70b-Instruct-hf", 2: "WizardLM/WizardCoder-Python-34B-V1.0"}
    label2id = {v: k for k, v in id2label.items()}
    training_args = TrainingArguments(
        output_dir="./finetuned_models",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        evaluation_strategy="epoch",
        save_total_limit=2,
        no_cuda=True,
        gradient_accumulation_steps=4,
    )
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3, id2label=id2label, label2id=label2id)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    print("finished setup, starting training\n")
    trainer.train()
    trainer.save_model("./finetuned_models")

# Dataset formatting instruction - https://docs.together.ai/docs/fine-tuning-data-preparation
def finetune_engineer(dataset_path):
    file_id = CLIENT.files.upload(file=dataset_path)["id"] # uploads data file to together api
    resp = CLIENT.fine_tuning.create(
        training_file = file_id,
        model = "meta-llama/Meta-Llama-3-8B",
        n_epochs = 3,
        n_checkpoints = 1,
        batch_size = 4,
        learning_rate = 1e-5,
    )