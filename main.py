from llm import LLMWrapper
import ast

def control_loop(organizer: LLMWrapper, swarm: dict):
    print("Hello, and welcome to Hivemind! This program will continually ask for input and respond. Whenever you want to quit, simply enter \'exit\'.\n")
    while(True):
        inp = input("Please enter query: ")
        if inp == "exit":
            break
        intermediate = organizer.generateChatResponse(content=inp)
        try:
            parsed_inter = ast.literal_eval(intermediate)
        except:
            print("Intermediate product failed to produce a dict, aborting.")
            continue
        try:
            final = swarm[parsed_inter["model"]].generateChatResponse(content=parsed_inter["prompt-engineered-context"] + parsed_inter["rephrased-prompt"])
        except:
            print("Intermediate product failed to produce correct dict keys, aborting.")
            continue
        print("\nFinal Response:" + final + "\n")
    print("Goodbye!")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    organizer_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    api_key = input("Please enter API key to continue: ")
    hives = {
        "mistralai/Mixtral-8x7B-Instruct-v0.1": LLMWrapper(model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=api_key, role="hive")
    }
    organizer = LLMWrapper(model=organizer_model, api_key=api_key, role="organizer", hive_workers=hives)
    control_loop(organizer, hives)

