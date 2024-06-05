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
genai.configure(api_key=GEMINI_API_KEY)

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
def finetune_classifier(dataset_path, cpu_or_gpu):
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
    if cpu_or_gpu == "cpu":
        training_args = TrainingArguments(
            output_dir="./finetuned_models",
            learning_rate=3e-5,  # Conservative learning rate
            per_device_train_batch_size=4,  # Smaller batch size to fit in CPU memory
            per_device_eval_batch_size=4,  # Matching the train batch size
            num_train_epochs=3,  # More epochs to compensate for smaller batch size
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,  # Usually disabled for initial runs
            logging_dir='./logs_cpu',  # Directory for storing logs
            logging_steps=200,  # Less frequent logging to reduce overhead
            evaluation_strategy="epoch",  # Evaluate at the end of each epoch
            save_total_limit=2,  # Keep only the last 2 checkpoints
            no_cuda=True,  # Disable CUDA (GPU) usage
            gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
        )
    if cpu_or_gpu == "gpu":
        training_args = TrainingArguments(
            output_dir="./finetuned_models",
            learning_rate=3e-5,  # Adjust as necessary
            per_device_train_batch_size=16,  # Larger batch size for GPU
            per_device_eval_batch_size=16,  # Matching the train batch size
            num_train_epochs=3,  # Sufficient epochs for fine-tuning
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,  # Usually disabled for initial runs
            logging_dir='./logs_gpu',  # Directory for storing logs
            logging_steps=100,  # More frequent logging
            evaluation_strategy="steps",  # More frequent evaluation
            eval_steps=500,  # Evaluate every 500 steps
            save_steps=500,  # Save checkpoint every 500 steps
            save_total_limit=3,  # Keep only the last 3 checkpoints
            warmup_steps=500,  # Warm-up learning rate for the first 500 steps
            fp16=True,  # Use mixed precision training
            gradient_accumulation_steps=2,  # Accumulate gradients to simulate larger batch size
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
    ) # starts together api finetuning job

#finetune_classifier("./training/first_training_set/combined_training.json", "cpu")
"""
{'eval_loss': 1.042781114578247, 'eval_accuracy': 0.45, 'eval_runtime': 7.7288, 'eval_samples_per_second': 2.588, 'eval_steps_per_second': 0.647, 'epoch': 2.0}    
{'eval_loss': 1.0311684608459473, 'eval_accuracy': 0.45, 'eval_runtime': 7.031, 'eval_samples_per_second': 2.845, 'eval_steps_per_second': 0.711, 'epoch': 3.0}    
{'train_runtime': 370.8887, 'train_samples_per_second': 0.647, 'train_steps_per_second': 0.04, 'train_loss': 1.0845155080159505, 'epoch': 3.0}                     
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 15/15 [06:10<00:00, 24.73s/it]
"""

# CLASSIFIER = {"model": BertForSequenceClassification.from_pretrained("./finetuned_models"), 
#               "tokenizer": BertTokenizer.from_pretrained("./finetuned_models")}
# print(classify_with_organizer("Create a Python program to generate a series of random numbers based on a user input."))

print(improve_prompt("Tell me fun things to do in new york"))

"""
Current plan:
0. Finalize list of models to make available and choose from (preferably ~2 code generation and ~2 chat)
1. Automate output and labelling for classifier model dataset to scale to 1k-2k datapoints (side by side model comparisons w/ LLM evals)
2. Transition classifier finetuning to GPU, increase training argument intensity
3. Build up dataset for prompt engineering (1k-2k datapoints) (have written by llama 3 70b OR GPT4)
4. Setup together api finetuning job
5. Do side by side eval comparisons of passing normal prompt to an LLM and passing prompt-engineered
6. Compare general input/output of architecture vs llama-3-8b

Finetune the same model on different datasets to create experts
Finetune a classifier to distinguish between them

Finetune on corpuses of raw data (auxillary - especially if we want to compare rigorously)
Finetune classifier on listwise scoring of models side by side
Evaluate on MMLU - use an API to check whether answer is correct or not (track which end models are chosen as perentages + compute individual model performance)

"""