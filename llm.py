import os
from together import Together
from pydantic import BaseModel, Field

# Prompt engineering context for organizer model to most effectively transform and format intermediate product.
ORGANIZER_PROMPT_CONTEXT = """
Given the following prompt, please fill in each json category with the following:
1. Produce any prompt engineering context you feel is useful and state it in as the prompt-engineered context.
2. Choose a model that will most accurately and holistically answer the prompt. Please choose from the provided list.
3. Rephrase the prompt to be answered most effectively by a large language model like ChatGPT or llama2. It's
important that I get the best results back (no hallucinations, lots of real information).

Remember that whatever you output will be immediately passed to a large language model, so be careful in your formatting. Additionally,
your only job is to classify and rephrase the input, do not answer any part of it. Take a minute to breath, and do your best.\n
"""

# Dict of available worker models, conforming to the standard "model name" : "description"
WORKER_DICT = {"meta-llama/Llama-3-70b-chat-hf": "Llama 3 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.",
               "codellama/CodeLlama-70b-Python-hf": "Code Llama is a family of large language models for code based on Llama 2 providing infilling capabilities, support for large input contexts, and zero-shot instruction following ability for programming tasks.",
               "Meta-Llama/Llama-Guard-7b": "Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations",
               "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": "Nous Hermes 2 Mixtral 7bx8 DPO is the new flagship Nous Research model trained over the Mixtral 7bx8 MoE LLM. The model was trained on over 1,000,000 entries of primarily GPT-4 generated data, as well as other high quality data from open datasets across the AI landscape, achieving state of the art performance on a variety of tasks.",
               "mistralai/Mixtral-8x7B-Instruct-v0.1": "The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts."}

class Organizer(BaseModel):
    prompt_engineered_context: str = Field(description="prompt-engineered-context")
    model: str = Field(description="model")
    rephrased_prompt: str = Field(description="rephrased-prompt")

class LLMWrapper:
    def __init__(self, model, api_key, role, hive_workers = {}):
        self.model_name = model
        self.client = Together(api_key=api_key)
        self.role = role
        # only non-empty if organizer
        self.hive_workers = hive_workers
    
    # Assumes organizer model is capable of json mode
    def generateChatResponse(self, content):
        if self.role == "organizer":
            content = ORGANIZER_PROMPT_CONTEXT + "Models available for task:" + str(self.hive_workers) + "\nPrompt:" + content
            response_format = {"type": "json_object", "schema": Organizer.model_json_schema()}
            messages = [{"role": "system", "content": "You are an organizer who formats your outputs in JSON. The JSON keys MUST be: 'prompt-engineered-context', 'model', 'rephrased-prompt'. Never make any other keys."}, {"role": "user", "content": content}]
            # Potentially return multiple responses here to provide more context for future models to take advantage of?
            response = self.client.chat.completions.create(model=self.model_name, response_format=response_format, messages=messages)
        else:
            response = self.client.chat.completions.create(model=self.model_name, messages=[{"role": "user", "content": content}])
        return response.choices[0].message.content
