import ollama

# Prompt engineering context for organizer model to most effectively transform and format intermediate product.
# TODO: replace this with actual prompt engineering process
ORGANIZER_PROMPT_CONTEXT = """
Given the following prompt, please fill in each json category with the following:
1. Produce any prompt engineering context you feel is useful and state it in as the prompt-engineered context.
2. Choose a model that will most accurately and holistically answer the prompt. Please choose from the provided list.
3. Rephrase the prompt to be answered most effectively by a large language model like ChatGPT or llama2. It's
important that I get the best results back (no hallucinations, lots of real information).

Remember that whatever you output will be immediately passed to a large language model, so be careful in your formatting. Additionally,
your only job is to classify and rephrase the input, do not answer any part of it. Take a minute to breath, and do your best.\n
"""

class LLMWrapper:
    def __init__(self, model, role, hive_workers = {}):
        self.model_name = model
        self.role = role
        # only non-empty if organizer
        self.hive_workers = hive_workers
    
    # Assumes organizer model is capable of json mode
    def generateChatResponse(self, system, content):
        if self.role == "organizer":
            content = ORGANIZER_PROMPT_CONTEXT + "Models available for task:" + str(self.hive_workers) + "\nPrompt:" + content
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        return ollama.chat(model=self.model_name, messages=messages)['message']['content']
