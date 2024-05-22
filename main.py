from llm import LLMWrapper
import re
import json

ORGANIZER_SYS_STATEMENT = """You are an organizer who formats your outputs in JSON. The JSON keys MUST be: 'prompt-engineered-context', 'model', 'rephrased-prompt'. Never make any other keys."""

def control_loop(organizer: LLMWrapper, swarm: dict):
    print("Hello, and welcome to Hivemind! This program will continually ask for input and respond. Whenever you want to quit, simply enter \'exit\'.\n")
    while(True):
        inp = input("Please enter query: ")
        if inp == "exit":
            break
        # fetches input, removes everything outside of {}
        inter_1 = re.search(r'\{.*\}', organizer.generateChatResponse(system=ORGANIZER_SYS_STATEMENT, content=inp), re.DOTALL).group(0).strip()
        # removes weird escape characters
        inter_2 = re.sub(r'[^\x20-\x7E]', '', inter_1)
        # formats as string
        intermediate = f"""{inter_2}"""
        try:
            # converts to dictionary
            parsed_inter = json.loads(intermediate)
        except:
            print("Intermediate product failed to produce a dict, aborting.")
            print(intermediate)
            continue
        try:
            # passes the updated prompt and context to final model
            # swarm[parsed_inter["model"]] is the chosen model
            final = swarm[parsed_inter["model"]].generateChatResponse(system=parsed_inter["prompt-engineered-context"], content=parsed_inter["rephrased-prompt"])
        except:
            print("Intermediate product failed to produce correct dict keys, aborting.")
            print(intermediate)
            continue
        print("\nFinal Response:" + final + "\n")
    print("Goodbye!")

def evaluate(organizer: LLMWrapper, eval_inputs: str, eval_answers: str, line_count: int):
    inp_f = open(eval_inputs, "r")
    ans_f = open(eval_answers, "r")
    total = 0
    for i in range(line_count):
        print("Run " + str(i))
        inp = inp_f.readline()
        answer = ans_f.readline().strip()
        # fetches input, removes everything outside of {}
        try:
            inter_1 = re.search(r'\{.*\}', organizer.generateChatResponse(system=ORGANIZER_SYS_STATEMENT, content=inp), re.DOTALL).group(0).strip()
        except:
            print("Model response doesn't include json.")
            print("Ratio so far: " + str(total/line_count))
            continue
        # removes weird escape characters
        inter_2 = re.sub(r'[^\x20-\x7E]', '', inter_1)
        # formats as string
        intermediate = f"""{inter_2}"""
        try:
            # converts to dictionary
            parsed_inter = json.loads(intermediate)
        except:
            print("Intermediate product failed to produce a dict, aborting.")
            print(intermediate)
            continue
        try:
            print(parsed_inter["model"])
            total += 1 if parsed_inter["model"].lower().replace(" ", "") == answer else 0
        except:
            print("Model response malformed.")
        print("Ratio so far: " + str(total/line_count))
    return total/line_count

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    organizer_model = "llama3"
    hives = {
        "llama3": LLMWrapper(model="llama3", role="hive"),
        "codegemma": LLMWrapper(model="codegemma", role="hive"),
        "llama2-chinese": LLMWrapper(model="llama2-chinese", role="hive")
    }
    # Keep a list of the models in our hivemind and their respective descriptions—useful for prompt engineering
    descriptions = {"llama3": "llama3 excels in generating coherent and \
                            context-specific text, demonstrating a strong understanding of language \
                            and context, making it suitable for conversational AI, question answering , \
                            and content creation. However, it may generate biased or toxic content, \
                            struggle with common sense and real-world knowledge, and produce responses \
                            that lack creativity and originality, are factually incorrect, or repetitive.", 

                            "llama2-chinese": "llama2-chinese excels in understanding and generating \
                            Chinese text, demonstrating strong language comprehension and fluency, \
                            making it suitable for Chinese language-related tasks such as conversation, \
                            question answering, and text generation. However, it may struggle with nuanced or \
                            idiomatic expressions, generate responses that lack cultural sensitivity or context,\
                            and be prone to errors in tone, style, or grammar, particularly when dealing with \
                            complex or specialized topics.", 
                            
                            "codegemma": "codegemma excels in generating high-quality code snippets and solutions \
                            in various programming languages, demonstrating strong coding knowledge and problem-solving \
                            abilities, making it suitable for tasks such as code completion, bug fixing, and programming \
                            assistance. However, it may struggle with complex or highly abstract problems, generate code \
                            that lacks readability or maintainability, and be prone to errors or inconsistencies,\
                            particularly when dealing with novel or edge cases."

                            }
    organizer = LLMWrapper(model=organizer_model, role="organizer", hive_workers=hives, hive_descriptions=descriptions)
    #control_loop(organizer, hives)
    print("Final ratio: " + str(evaluate(organizer=organizer, eval_inputs="training_dataset/evaluation.txt", eval_answers="training_dataset/evaluation_answers.txt", line_count=500)))

