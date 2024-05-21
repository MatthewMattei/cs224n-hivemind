# Right now, these prompt engineering functions JUST generates which model each input should be
# passed to. It does not generate prompt engineering context or rephrase the prompt.

# Keep a list of the models in our hivemind and their respective descriptions—useful for prompt engineering
models_with_descriptions = {"llama3": "LLaMA-3 excels in generating coherent and \
                            context-specific text, demonstrating a strong understanding of language \
                            and context, making it suitable for conversational AI, question answering , \
                            and content creation. However, it may generate biased or toxic content, \
                            struggle with common sense and real-world knowledge, and produce responses \
                            that lack creativity and originality, are factually incorrect, or repetitive.", 

                            "llama2-chinese": "LLaMA-2-Chinese excels in understanding and generating \
                            Chinese text, demonstrating strong language comprehension and fluency, \
                            making it suitable for Chinese language-related tasks such as conversation, \
                            question answering, and text generation. However, it may struggle with nuanced or \
                            idiomatic expressions, generate responses that lack cultural sensitivity or context,\
                            and be prone to errors in tone, style, or grammar, particularly when dealing with \
                            complex or specialized topics.", 
                            
                            "codegemma": "CodeGEMMA excels in generating high-quality code snippets and solutions \
                            in various programming languages, demonstrating strong coding knowledge and problem-solving \
                            abilities, making it suitable for tasks such as code completion, bug fixing, and programming \
                            assistance. However, it may struggle with complex or highly abstract problems, generate code \
                            that lacks readability or maintainability, and be prone to errors or inconsistencies,\
                            particularly when dealing with novel or edge cases."

                            }

# Few-shot prompting: providing examples alongside our input. This function will provide a list of example inputs
# for each of the respective models in our hivemind so our queen bee has an idea of what inputs should be passed to
# each model
def few_shot_prompting(input, models_with_descriptions):
    few_shot_context = """You will be provided with a list of models and you must determine which one an input should be passed to. Here are 
    some examples of inputs and the respective models they should be passed to:
    Input: What is the capital of France?
    Corresponding Model: llama3

    Input: What is your favorite Chinese food?
    Corresponding Model: llama3

    Input: 我喜欢苹果
    Corresponding Model: llama2-chinese

    Input: 我想吃饭了，你能够告诉一个很好吃的的放吗？
    Corresponding Model: llama2-chinese

    Input: Write a Python function to calculate the factorial of a number
    Corresponding Model: codegemma

    Input: How do I implement a binary search algorithm in Java?
    Corresponding Model: codegemma

    Examine this list of models and their respective descriptions: """ + str(models_with_descriptions) + """. Determine which should be utilized for this input: """ + str(input)
    + """Your answer should only return the name of the chosen model and nothing else"""
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
    Corresponding Model: llama3

    Input: 我想吃饭了，你能够告诉一个很好吃的的放吗？
    Thought Process: This is a Chinese-language input asking for restaurant recommendations. This should be passed to
    llama2-chinese because that model takes Chinese inputs and is good at question answering.
    Corresponding Model: llama2-chinese

    Input: Write a Python function to calculate the factorial of a number
    Thought Process: This is an English-language input requesting code. This should be passed to codegemma because codegemma
    is good at generating code.
    Corresponding Model: codegemma

    Examine this list of models and their respective descriptions: """ + str(models_with_descriptions) + """. Using similar thought process to the examples, determine
    which should be utilized for this input: """ + str(input) + """Your answer should only return the name of the chosen 
    model and nothing else."""
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
    
