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
        inter_1 = re.search(r'\{.*\}', organizer.generateChatResponse(system=ORGANIZER_SYS_STATEMENT, content=inp), re.DOTALL).group(0).strip()
        inter_2 = re.sub(r'[^\x20-\x7E]', '', inter_1)
        intermediate = f"""{inter_2}"""
        try:
            parsed_inter = json.loads(intermediate)
        except:
            print("Intermediate product failed to produce a dict, aborting.")
            print(intermediate)
            continue
        try:
            final = swarm[parsed_inter["model"]].generateChatResponse(system=parsed_inter["prompt-engineered-context"], content=parsed_inter["rephrased-prompt"])
        except:
            print("Intermediate product failed to produce correct dict keys, aborting.")
            print(intermediate)
            continue
        print("\nFinal Response:" + final + "\n")
    print("Goodbye!")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    organizer_model = "llama3"
    hives = {
        "llama3": LLMWrapper(model="llama3", role="hive"),
        "mistral": LLMWrapper(model="mistral", role="hive"),
        "llama2": LLMWrapper(model="mistral", role="hive")
    }
    organizer = LLMWrapper(model=organizer_model, role="organizer", hive_workers=hives)
    control_loop(organizer, hives)

