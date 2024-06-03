import torch
import math

# This function takes in a list called input_ranking that's our ranked list of models by performance, as well as a list called
# ground_truth_ranking that is the correct ranked list of models by performance. It returns a tensor output where output[i]
# is the difference between i and the index of ground_truth_ranking[i] in input_ranking
def obtain_numerical_difference(ground_truth_ranking, input_ranking):
    if len(ground_truth_ranking) != len(input_ranking):
        print("ground_truth_ranking: " + str(ground_truth_ranking) + " input_ranking: " + str(input_ranking))
        raise Exception("Ground truth list and input list are not the same size")
    output = torch.zeros(len(ground_truth_ranking))
    for i in range(len(ground_truth_ranking)):
        output[i] = abs(input_ranking.index(ground_truth_ranking[i]) - i)
    return output

# This function is designed to take the output from obtain_numerical_difference and compute the loss based on that. It computes
# a mean squared loss but with an additional step: after squaring each of the entries, it divides by the square root of the index
# of the entry. This weighs the top items more heavily, which is what we want because we care more about predicting the best 
# model than the 2nd best, 3rd best, etc. The average is taken over these values to return the loss, which is a float.
def compute_loss(differences):
    differences = differences.to(torch.float32) # maybe not necessary, but here just in case
    differences = torch.square(differences)
    print(differences)
    for i in range((differences.size())[0]):
        if (differences[i] != 0):
            differences[i] = differences[i]/math.sqrt(i+1)
    print(differences)
    mean_squared_loss = torch.mean(differences)
    return float(mean_squared_loss)


# testing code
list1 = ["apples", "bananas", "mangos", "pears"]
list2 = ["pears", "bananas", "apples", "mangos"]
# numerical difference: [2, 0, 1, 3]
# loss: 2.2693
numerical_difference = obtain_numerical_difference(list1, list2)
print(numerical_difference)
loss = compute_loss(numerical_difference)
print(loss)