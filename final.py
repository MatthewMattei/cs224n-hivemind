# Imports
import os
from together import Together
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TOGETHER_API_KEY = "af55a5d60e08e7064287b3099b7c22c18366a4bee70bcc4e25beb839a40ce8c2"

CLIENT = Together(api_key=TOGETHER_API_KEY)

CLASSIFIER = {"model": BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased"), 
              "tokenizer": BertTokenizer.from_pretrained("google-bert/bert-base-uncased")}

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

# Dataset formatting instructions - https://huggingface.co/docs/transformers/en/tasks/sequence_classification#train
def finetune_classifier(dataset_path, cpu_or_gpu):
    def load_dataset_from_csv(file_path):
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)

    def tokenize_dataset(dataset, tokenizer):
        def preprocess_function(examples):
            return tokenizer(examples['text'], padding='max_length', truncation=True)
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        return tokenized_dataset
    
    dataset = load_dataset_from_csv(dataset_path)
    label_encoder = LabelEncoder()
    dataset = dataset.map(lambda examples: {'label': label_encoder.fit_transform(examples['label'])})
    dataset = dataset.train_test_split(test_size=0.2)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
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
    num_labels = len(label_encoder.classes_)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
    )
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