import matplotlib.pyplot as plt
import numpy as np
import textwrap
import math

barWidth = 0.2

# Finetuned Models - Subject-Based Accuracy

plt.figure(1)
fig = plt.subplots(figsize = (8, 6))

humanities_model = [176, 153, 171, 153]
other_model = [178, 150, 159, 155]
ss_model = [179, 145, 172, 156]
stem_model = [179, 145, 172, 156]

br1 = np.arange(len(humanities_model))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]

plt.bar(br1, humanities_model, color = 'r', width = barWidth, label = "Finetuned Humanities Model")
plt.bar(br2, other_model, color = 'b', width = barWidth, label = "Finetuned Other Model")
plt.bar(br3, ss_model, color = 'g', width = barWidth, label = "Finetuned Social Sciences Model")
plt.bar(br4, stem_model, color = 'y', width = barWidth, label = "Finetuned STEM Model")

labels = [
    textwrap.fill('STEM', 20),
    textwrap.fill('Humanities', 20),
    textwrap.fill('Social Sciences', 20),
    textwrap.fill('Other', 20)
]
plt.xlabel('Categories')
plt.ylabel('Amount Correct')
plt.title("Finetuned Models - Subject-Based Accuracy")
plt.xticks([r + 1.5 * barWidth for r in range(len(humanities_model))], labels)
plt.ylim(top=250) 
plt.legend()
plt.savefig("figures/Finetuned Models - Subject Based")

# Finetuned Models - Total Accuracy

plt.figure(2)
fig = plt.subplots(figsize = (8, 6))

labels = [
    textwrap.fill('Finetuned Humanities Model', 20),
    textwrap.fill('Finetuned Other Model', 20),
    textwrap.fill('Finetuned Social Sciences Model', 20),
    textwrap.fill('Finetuned STEM Model', 20)
]
values = [653, 642, 652, 627]
colors = ['r', 'b', 'g', 'y']

bars = plt.bar(labels, values, color=colors, width=0.4)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, yval, ha='center', va='bottom')

plt.xlabel('Finetuned Models')
plt.ylabel('Amount Correct')
plt.title("Finetuned Models - Total Accuracy")
plt.xticks([r for r in range(len(humanities_model))], labels)
plt.ylim(top=1000)
plt.savefig("figures/Finetuned Models - Total")

# Baseline Model & Mega-Model - Subject-Based Accuracy

plt.figure(3)
fig = plt.subplots(figsize = (8, 6))

baseline_model = [104, 67, 89, 58]
mega_model = [178, 149, 170, 157]

br1 = np.arange(len(baseline_model))
br2 = [x + barWidth for x in br1]

plt.bar(br1, baseline_model, color = 'r', width = barWidth, label = "Baseline Model")
plt.bar(br2, mega_model, color = 'b', width = barWidth, label = "Mega-Model")

labels = [
    textwrap.fill('STEM', 20),
    textwrap.fill('Humanities', 20),
    textwrap.fill('Social Sciences', 20),
    textwrap.fill('Other', 20),
]
plt.xlabel('Categories')
plt.ylabel('Amount Correct')
plt.title("Baseline Model & Mega-Model - Subject-Based Accuracy")
plt.xticks([r + barWidth / 2 for r in range(len(humanities_model))], labels)
plt.ylim(top=250) 
plt.legend()
plt.savefig("figures/Baseline Model and Mega-Model - Subject-Based Accuracy")

# Baseline Model, Mega-Model, Classifier - Total Accuracy

plt.figure(4)
fig = plt.subplots(figsize = (8, 6))

labels = [
    textwrap.fill('Baseline Model', 20),
    textwrap.fill('Mega-Model', 20),
    textwrap.fill("Classifier", 20)
]
values = [318, 654, 642]
colors = ['r', 'b', 'g']

bars = plt.bar(labels, values, color=colors, width=0.5)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, yval, ha='center', va='bottom')

plt.xlabel('Models')
plt.ylabel('Amount Correct')
plt.title("Baseline Model, Mega-Model, Classifier - Total Accuracy")
plt.xticks([r for r in range(len(values))], labels)
plt.ylim(top=1000)
plt.savefig("figures/Baseline Model, Mega-Model, Classifier - Total")

# Classifier - Subject-Based Accuracy

plt.figure(5)
fig = plt.subplots(figsize = (8, 6))

subject_total = [75, 161, 0, 764]
subject_correct = [57, 90, 0, 495]

color_total = ['lightcoral', 'powderblue', 'palegreen', 'palegoldenrod']
color_subject = ['firebrick', 'b', 'g', 'gold']

br1 = np.arange(len(baseline_model))

bars = plt.bar(br1, subject_total, color = color_total, width = barWidth * 2, label = "Total Inputs")
plt.bar(br1, subject_correct, color = color_subject, width = barWidth * 2, label = "Correct Inputs")

for i in range(len(bars)):
    bar = bars[i]
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, str(math.trunc(subject_correct[i]/max(1, subject_total[i]) * 100)) + "%", ha='center', va='bottom')

labels = [
    textwrap.fill('STEM', 20),
    textwrap.fill('Humanities', 20),
    textwrap.fill('Social Sciences', 20),
    textwrap.fill('Other', 20),
]
plt.xlabel('Categories')
plt.ylabel('Amount Correct')
plt.title("Classifier - Subject-Based Accuracy")
plt.xticks([r for r in range(len(humanities_model))], labels)
plt.ylim(top=1000)
plt.savefig("figures/Classifier - Subject-Based Accuracy")

# Perfect Pipeline - Subject-Based Accuracy

plt.figure(6)
fig = plt.subplots(figsize = (8, 6))

subject_total = [250, 250, 250, 250]
subject_correct = [173, 153, 172, 155]

color_total = ['lightcoral', 'powderblue', 'palegreen', 'palegoldenrod']
color_subject = ['firebrick', 'b', 'darkgreen', 'gold']

br1 = np.arange(len(baseline_model))

bars = plt.bar(br1, subject_total, color = color_total, width = barWidth * 2, label = "Total Inputs")
plt.bar(br1, subject_correct, color = color_subject, width = barWidth * 2, label = "Correct Inputs")

for i in range(len(bars)):
    bar = bars[i]
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, str(math.trunc(subject_correct[i]/max(1, subject_total[i]) * 100)) + "%", ha='center', va='bottom')

labels = [
    textwrap.fill('STEM', 20),
    textwrap.fill('Humanities', 20),
    textwrap.fill('Social Sciences', 20),
    textwrap.fill('Other', 20),
]
plt.xlabel('Categories')
plt.ylabel('Amount Correct')
plt.title("Perfect Pipeline - Subject-Based Accuracy")
plt.xticks([r for r in range(len(humanities_model))], labels)
plt.ylim(top=300)
plt.savefig("figures/Perfect Pipeline - Subject-Based Accuracy")

# Random Classifier, Random Majority Classifier, Majority Classifier - Subject-Based Accuracy

plt.figure(7)
fig = plt.subplots(figsize = (8, 6))

random_classifier_correct = [179, 152, 161, 150]
random_classifier_total = [279, 219, 254, 248]
random_majority_correct = [165, 153, 189, 149]
random_majority_total = [260, 231, 276, 233]
majority_correct = [603, 31, 7, 4]
majority_total = [883, 69, 32, 16]

br1 = np.arange(len(random_classifier_correct))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

plt.bar(br1, random_classifier_total, color = 'lightcoral', width = barWidth)
plt.bar(br1, random_classifier_correct, color = 'firebrick', width = barWidth, label = "Random Classifier")
plt.bar(br2, random_majority_total, color = 'powderblue', width = barWidth)
plt.bar(br2, random_majority_correct, color = 'b', width = barWidth, label = "Random Majority Classifier")
plt.bar(br3, majority_total, color = 'palegreen', width = barWidth)
plt.bar(br3, majority_correct, color = 'g', width = barWidth, label = "Majority Classifier")

labels = [
    textwrap.fill('STEM', 20),
    textwrap.fill('Humanities', 20),
    textwrap.fill('Social Sciences', 20),
    textwrap.fill('Other', 20)
]
plt.xlabel('Categories')
plt.ylabel('Amount Correct')
plt.title("Random Classifier, Random Majority Classifier, Majority Classifier - Subject-Based Accuracy")
plt.xticks([r + 1.5 * barWidth for r in range(len(humanities_model))], labels)
plt.ylim(top=1000) 
plt.legend()
plt.savefig("figures/Random Figures")

# Random Classifier, Random Majority Classifier, Majority Classifier - Total Accuracy

plt.figure(8)
fig = plt.subplots(figsize = (8, 6))

labels = [
    textwrap.fill('Random Classifier', 20),
    textwrap.fill('Random Majority Classifier', 20),
    textwrap.fill("Majority Classifier", 20)
]
values = [642, 656, 645]
colors = ['r', 'b', 'g']

bars = plt.bar(labels, values, color=colors, width=0.5)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, yval, ha='center', va='bottom')

plt.xlabel('Models')
plt.ylabel('Amount Correct')
plt.title("Random Classifier, Random Majority Classifier, Majority Classifier - Total Accuracy")
plt.xticks([r for r in range(len(values))], labels)
plt.ylim(top=1000)
plt.savefig("figures/Random Figures - Total")

# Classifier 2.0 - Subject-Based Accuracy

plt.figure(9)
fig = plt.subplots(figsize = (8, 6))

subject_total = [334, 333, 333]
subject_correct = [183, 296, 93]

color_total = ['lightcoral', 'powderblue', 'palegreen']
color_subject = ['firebrick', 'b', 'g']

br1 = np.arange(len(subject_total))

bars = plt.bar(br1, subject_total, color = color_total, width = barWidth * 2, label = "Total Inputs")
plt.bar(br1, subject_correct, color = color_subject, width = barWidth * 2, label = "Correct Inputs")

for i in range(len(bars)):
    bar = bars[i]
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 5, str(math.trunc(subject_correct[i]/max(1, subject_total[i]) * 100)) + "%", ha='center', va='bottom')

labels = [
    textwrap.fill('STEM', 20),
    textwrap.fill('Humanities', 20),
    textwrap.fill('Social Sciences', 20),
]
plt.xlabel('Categories')
plt.ylabel('Amount Correct')
plt.title("Classifier 2.0 - Subject-Based Accuracy")
plt.xticks([r for r in range(len(subject_total))], labels)
plt.ylim(top=400)
plt.savefig("figures/Classifier 2 - Subject-Based Accuracy")