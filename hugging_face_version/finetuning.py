# changing run.py to work for us, most of this is scrap work rn
import os
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader


#Much of this code borrowed or adapted from https://huggingface.co/docs/transformers/en/training

# Determine the device to use (GPU, MPS, or CPU)
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
device = get_device()

organizer_model = AutoModelForSeq2SeqLM.from_pretrained("google-bert/bert-base-uncased").to(device)
organizer_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

dataset = load_dataset("OpenOrca")
def tokenize_function(examples):
    return organizer_tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)
tokenized_dataset = tokenized_dataset.remove_columns(["id", "system_prompt", "question"])
tokenized_dataset = tokenized_dataset.rename_column(["response", "labels"])
tokenized_dataset.setformat("torch")
small_train_dataset = tokenized_dataset["train"].shuffle(seed=69).select(range(500))
small_eval_dataset = tokenized_dataset["train"].shuffle(seed=69).select(range(100))
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=4)
eval_dataloader = DataLoader(small_eval_dataset, shuffle=True, batch_size=4)


def finetune_organizer_model(organizer_model, organizer_tokenizer, small_train_dataset, small_eval_dataset):
    optimizer = torch.optim.Adam(organizer_model.parameters(), lr=1e-5)
    
    for user_input in dataset:
        rephrased_input = rephrase_input(organizer_model, organizer_tokenizer, user_input) #need to make this
        
        ranked_models = rank_downstream_models(organizer_model, organizer_tokenizer, rephrased_input, downstream_models) # ramya is making this
        best_model, best_tokenizer = ranked_models[0]

        for model, tokenizer in ranked_models[:2]:
            inputs = tokenizer(rephrased_input, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Move inputs to CPU if using MPS
            if device.type == "mps":
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                model.to("cpu")
            
            output = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=inputs['input_ids'],  # Provide the input_ids as decoder_input_ids
                max_length=256,
                num_beams=5,
                early_stopping=True
            )
            
            # Move model back to the original device
            if device.type == "mps":
                model.to(device)
            
            reward = feedback_fn(output)
            rewards.append(reward)
        
        loss = compute_loss(rewards, ranked_models)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()