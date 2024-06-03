'''
File: evaluation.py
---------------------------------
Evaluation class for our training set. Evaluation currently done by accuracy

Format Info:
- The test_set is a list of test cases that may look like:
    - {{"user_response": "Hello!", "intermediary_model": "llama3"}, ...}
- organizer and swarm defined in main.py
'''
import json
import re
from typing import List, Dict

class Evaluation:
    def __init__(self, organizer: LLMWrapper, test_set: List[Dict[str, str]], swarm: dict):
        self.organizer = organizer
        self.swarm = swarm
        self.test_set = test_set

    # Utilize this function to evaluate given test_set and swarm
    def evaluation(self):
        accuracy_count = 0
        
        # Loops for every test case and increments accuracy as desired
        for test_case in self.test_set:
            user_response = test_case["user_response"]
            expected_intermediary_model = test_case["intermediary_model"]
            
            # @Matthew possibly need to check if this information can be accurately found from main.py?
            try:
                inter_1 = re.search(r'\{.*\}', self.organizer.generateChatResponse(system=ORGANIZER_SYS_STATEMENT, content=user_response), re.DOTALL).group(0).strip()
                inter_2 = re.sub(r'[^\x20-\x7E]', '', inter_1)
                intermediate = f"""{inter_2}"""
                parsed_inter = json.loads(intermediate)
                
                training_intermediary_model = self.swarm[parsed_inter["model"]].generateChatResponse(system=parsed_inter["prompt-engineered-context"], content=parsed_inter["rephrased-prompt"])
            except Exception as e:
                actual_intermediary_model = str(e)
            
            if self.evaluate_accuracy(expected_intermediary_model, actual_intermediary_model):
                accuracy_count += 1
        
        accuracy = accuracy_count / len(self.test_set)
        return accuracy

    # Compares if this strings are equal
    def evaluate_accuracy(self, expected_intermediary_model: str, actual_intermediary_model: str) -> bool:
        return expected_intermediary_model.strip().lower() == actual_intermediary_model.strip().lower()
