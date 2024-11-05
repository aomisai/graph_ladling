import re
import matplotlib.pyplot as plt

# Initialize variables to store configuration and data
dataset = ""
model_type = ""
eval_step = 10  # default value, will be overwritten if found in the input
epochs = []
test_accuracies = []
valid_accuracies = []
losses = []

# Read from the file
with open("experiment_log.txt", "r") as file:
    lines = file.readlines()

# Patterns to find dataset, model type, and eval step individually
dataset_pattern = re.compile(r"--dataset=(\S+)")
model_type_pattern = re.compile(r"--type_model=(\S+)")
eval_step_pattern = re.compile(r"--eval_steps=(\d+)")

# Look for the config parameters
for line in lines:
    if not dataset:
        dataset_match = dataset_pattern.search(line)
        if dataset_match:
            dataset = dataset_match.group(1)

    if not model_type:
        model_type_match = model_type_pattern.search(line)
        if model_type_match:
            model_type = model_type_match.group(1)

    if eval_step == 10:  # only update if still at the default
        eval_step_match = eval_step_pattern.search(line)
        if eval_step_match:
            eval_step = int(eval_step_match.group(1))

    # Stop searching once all values are found
    if dataset and model_type and eval_step != 10:
        break

# Pattern to match epoch log lines with relevant data
epoch_pattern = re.compile(r"Epoch: (\d+), Loss: ([\d.]+), Train: [\d.]+%, Valid: ([\d.]+)% Test: ([\d.]+)%")

# Parse the epoch data
for line in lines:
    if "Epoch" in line:
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            loss = float(epoch_match.group(2))
            valid_accuracy = float(epoch_match.group(3))
            test_accuracy = float(epoch_match.group(4))

            # Append data every `eval_step` epochs
            if epoch % eval_step == 0:
                epochs.append(epoch)
                losses.append(loss)
                valid_accuracies.append(valid_accuracy)
                test_accuracies.append(test_accuracy)

# Plot Validation and Test Accuracies in one graph
plt.figure(figsize=(10, 5))
plt.plot(epochs, valid_accuracies, label="Validation Accuracy", marker='o', color='green')
plt.plot(epochs, test_accuracies, label="Test Accuracy", marker='o', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title(f"{model_type} Model on {dataset} - Validation and Test Accuracy per Epoch")
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss in a separate graph
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, label="Loss", marker='o', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{model_type} Model on {dataset} - Loss per Epoch")
plt.legend()
plt.grid(True)
plt.show()
