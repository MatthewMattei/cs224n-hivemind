# Right now, these prompt engineering functions JUST generates which model each input should be
# passed to. It does not generate prompt engineering context or rephrase the prompt.

# Few-shot prompting: providing examples alongside our input. This function will provide a list of example inputs
# for each of the respective models in our hivemind so our queen bee has an idea of what inputs should be passed to
# each model
def few_shot_prompting(input, models_with_descriptions):
    few_shot_context = """You will be provided with a list of models and you must determine which one will most accurately and holistically answer the prompt. 
    Here are some examples of inputs and the respective models that should be chosen:
    Input: What is the capital of France?
    Output: "llama3"

    Input: What is your favorite Chinese food?
    Output: "llama3"

    Input: 我喜欢苹果
    Output: "llama2-chinese"

    Input: 我想吃饭了，你能够告诉一个很好吃的的放吗？
    Output: "llama2-chinese"

    Input: Write a Python function to calculate the factorial of a number
    Output: "codegemma"

    Input: How do I implement a binary search algorithm in Java?
    Output: "codegemma"

    Here is a list of models you can choose from and their respective descriptions: \"""" + str(models_with_descriptions) + """\". With all of this in mind, 
    I am going to give you an input. Prompt: """ + str(input) + """Remember, this is to fill in the second Json category, the
    model selection. Make sure to fill in each Json category with the prompt engineering context, the model selection, and
    the prompt rephrasing."""
    return few_shot_context

#Chain of thought prompting is similar to few-shot prompting, but with the examples it provides it walks the LLM through its
#logic so it can build on that. A list of examples inputs are provided as well as a sample thought process and solution for each.
# The queen bee will mirror this logic when determining which model to send the actual input to. 
def chain_of_thought_prompting(input, models_with_descriptions):
    chain_of_thought_context = """You will be provided with a list of models and you must determine which one an input should be passed to. Here are 
    some examples of inputs and the respective models they should be passed to:

    Input: What is the capital of France?
    Thought Process: This is an English-language input asking a geography question that has a definitive answer. This should be passed to
    the llama3-model because llama3 excels at question answering.
    Output: llama3

    Input: 我想吃饭了，你能够告诉一个很好吃的的放吗？
    Thought Process: This is a Chinese-language input asking for restaurant recommendations. This should be passed to
    llama2-chinese because that model takes Chinese inputs and is good at question answering.
    Output: llama2-chinese

    Input: Write a Python function to calculate the factorial of a number
    Thought Process: This is an English-language input requesting code. This should be passed to codegemma because codegemma
    is good at generating code.
    Output: codegemma

    Here is a list of models you can choose from and their respective descriptions: """ + str(models_with_descriptions) + """. With all of this in mind, I am going to give
    you an input. Prompt: """ + str(input)
    return chain_of_thought_context

# Reflexion provides a type of self evaluation on an input. It provides a sample output, then generates an evaluation OF that output,
# then uses that evaluation to determine whether the original output needs to be modified or not. This way, the queen bee has a chance to
# provide herself with feedback and teach herself. This function is currently unfinished—it needs to be determined whether the
#output_1 and output_2 will be generated from the same model that everything else is running on or a DIFFERENT model altogether.
def reflexion_prompting(input, models_with_descriptions):
    output_1 = ""
    output_2 = ""
    input_1 = """You will be provided with a list of models and you must determine which one an input should be passed to. The list of models
    and their respective descriptions is """ + models_with_descriptions + """and the input is """ + input + """. Walk me through
    your thought process on why you're choosing the model you are. Make sure to specify both the original input and the
    model you're choosing in your """
    # run this in an LLM, store the output in output_1
    input_2 = """An LLM is attempting to determine which finetuned model an input should should be passed to to best optimize
    responses. The list of models and their descriptions is """ + models_with_descriptions + """The LLM's reasoning was""" + output_1 + """Provide 
    an evaluation on their thought process and see if it can be improved or refined, though don't make unnecessary criticisms."""
    # run this in an LLM, store the output in output_2
    input_3 = """An LLM is attempting to determine which finetuned model an input should should be passed to to best optimize
    responses. The list of models and their descriptions is """ + models_with_descriptions + """The LLM's reasoning was """ + output_1 + """An evaluation
    has been provided, it said: """ + output_2 + """Based on the original output and the evaluation provided, make a final decision on
    which model the original input should be passed to. You should return nothing but the final model"""
    return input_3
    
