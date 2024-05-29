# File path: organizer_model.py
import os
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np

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

# Load the organizer model and downstream models
def load_models(organizer_model_name, downstream_model_names):
    organizer_model = AutoModelForSeq2SeqLM.from_pretrained(organizer_model_name).to(device)
    organizer_tokenizer = AutoTokenizer.from_pretrained(organizer_model_name)

    downstream_models = []
    for model_name in downstream_model_names:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        downstream_models.append((model, tokenizer))
    
    return organizer_model, organizer_tokenizer, downstream_models

# Rephrase input using the organizer model
def rephrase_input(organizer_model, organizer_tokenizer, user_input):
    inputs = organizer_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Move inputs to CPU if using MPS
    if device.type == "mps":
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        organizer_model.to("cpu")
    
    outputs = organizer_model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=256,
        num_beams=5,
        early_stopping=True
    )
    
    # Move model back to the original device
    if device.type == "mps":
        organizer_model.to(device)
    
    rephrased_text = organizer_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return rephrased_text

# Choose the best downstream model
def choose_downstream_model(organizer_model, organizer_tokenizer, rephrased_input, downstream_models):
    inputs = organizer_tokenizer(rephrased_input, return_tensors="pt", padding=True, truncation=True).to(device)
    scores = []
    
    for model, tokenizer in downstream_models:
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=inputs['input_ids']  # Provide the input_ids as decoder_input_ids
            )
            scores.append(outputs.logits.mean().item())
    
    best_model_idx = np.argmax(scores)
    best_model, best_tokenizer = downstream_models[best_model_idx]
    return best_model, best_tokenizer

# Finetune the organizer model with reinforcement learning
def finetune_organizer_model(organizer_model, organizer_tokenizer, downstream_models, dataset, feedback_fn):
    optimizer = torch.optim.Adam(organizer_model.parameters(), lr=1e-5)
    
    for user_input in dataset:
        rephrased_input = rephrase_input(organizer_model, organizer_tokenizer, user_input)
        
        ranked_models = rank_downstream_models(organizer_model, organizer_tokenizer, rephrased_input, downstream_models)
        best_model, best_tokenizer = ranked_models[0]
        
        rewards = []
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

def rank_downstream_models(organizer_model, organizer_tokenizer, rephrased_input, downstream_models):
    inputs = organizer_tokenizer(rephrased_input, return_tensors="pt", padding=True, truncation=True).to(device)
    scores = []

    for model, tokenizer in downstream_models:
        with torch.no_grad():
            outputs = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                decoder_input_ids=inputs['input_ids']  # Provide the input_ids as decoder_input_ids
            )
            scores.append(outputs.logits.mean().item())

    ranked_indices = np.argsort(scores)[::-1]
    ranked_models = [downstream_models[i] for i in ranked_indices]
    return ranked_models

def compute_loss(rewards, ranked_models):
    loss = torch.tensor(0.0, requires_grad=True, device=device)  # Initialize loss as a tensor
    for i in range(len(rewards)):
        loss = loss + rewards[i] * (1 - i / len(ranked_models))  # Using torch.sum() instead of in-place addition
    return -loss


# Example usage
if __name__ == "__main__":
    organizer_model_name = "t5-base"
    downstream_model_names = ["facebook/bart-large-cnn", "google/pegasus-xsum", "t5-small"]
    
    organizer_model, organizer_tokenizer, downstream_models = load_models(organizer_model_name, downstream_model_names)
    
    user_input = "How do I implement a linked list in Python?"
    rephrased_input = rephrase_input(organizer_model, organizer_tokenizer, user_input)
    best_model, best_tokenizer = choose_downstream_model(organizer_model, organizer_tokenizer, rephrased_input, downstream_models)
    
    # Finetune with a dummy dataset and feedback function
    dummy_dataset = ["sample input 1", "sample input 2"]
    def dummy_feedback_fn(output): return np.random.rand()
    
    finetune_organizer_model(organizer_model, organizer_tokenizer, downstream_models, dummy_dataset, dummy_feedback_fn)
