import re
import matplotlib.pyplot as plt

# Initialize variables to store configuration and data
experiments = []
current_experiment = {
    "dataset": "",
    "model_type": "",
    "eval_step": 10,  # default value, will be overwritten if found in the input
    "seed": None,
    "epochs": [],
    "test_accuracies": [],
    "valid_accuracies": [],
    "losses": [],
    "best_valid": None,
    "best_test": None
}

# Patterns for configuration and data parsing
dataset_pattern = re.compile(r"--dataset=(\S+)")
model_type_pattern = re.compile(r"--type_model=(\S+)")
eval_step_pattern = re.compile(r"--eval_steps=(\d+)")
seed_pattern = re.compile(r"seed \(which_run\) = <(\d+)>")
epoch_pattern = re.compile(r"Epoch: (\d+), Loss: ([\d.]+), Train: [\d.]+%, Valid: ([\d.]+)% Test: ([\d.]+)%")
best_pattern = re.compile(r"Best train: [\d.]+%, Best valid: ([\d.]+)% Best test: ([\d.]+)%")

# Read from the file
with open("experiment_log.txt", "r") as file:
    lines = file.readlines()

# Parse lines
for line in lines:
    # Extract config data only if not set for the current experiment
    if not current_experiment["dataset"]:
        dataset_match = dataset_pattern.search(line)
        if dataset_match:
            current_experiment["dataset"] = dataset_match.group(1)

    if not current_experiment["model_type"]:
        model_type_match = model_type_pattern.search(line)
        if model_type_match:
            current_experiment["model_type"] = model_type_match.group(1)

    if current_experiment["eval_step"] == 10:  # only update if still at the default
        eval_step_match = eval_step_pattern.search(line)
        if eval_step_match:
            current_experiment["eval_step"] = int(eval_step_match.group(1))

    # Detect new experiment by seed
    seed_match = seed_pattern.search(line)
    if seed_match:
        if current_experiment["seed"] is not None:  # save current experiment before starting a new one
            experiments.append(current_experiment)
            current_experiment = {
                "dataset": current_experiment["dataset"],
                "model_type": current_experiment["model_type"],
                "eval_step": current_experiment["eval_step"],
                "seed": int(seed_match.group(1)),
                "epochs": [],
                "test_accuracies": [],
                "valid_accuracies": [],
                "losses": [],
                "best_valid": None,
                "best_test": None
            }
        else:
            current_experiment["seed"] = int(seed_match.group(1))

    # Extract epoch data
    epoch_match = epoch_pattern.search(line)
    if epoch_match:
        epoch = int(epoch_match.group(1))
        loss = float(epoch_match.group(2))
        valid_accuracy = float(epoch_match.group(3))
        test_accuracy = float(epoch_match.group(4))

        # Append data every `eval_step` epochs
        if epoch % current_experiment["eval_step"] == 0:
            current_experiment["epochs"].append(epoch)
            current_experiment["losses"].append(loss)
            current_experiment["valid_accuracies"].append(valid_accuracy)
            current_experiment["test_accuracies"].append(test_accuracy)

    # Extract best validation and test accuracy at the end of each experiment
    best_match = best_pattern.search(line)
    if best_match:
        current_experiment["best_valid"] = float(best_match.group(1))
        current_experiment["best_test"] = float(best_match.group(2))

# Append the last experiment after finishing the loop
if current_experiment["seed"] is not None:
    experiments.append(current_experiment)

# Display options for user to select an experiment
print("Available experiments:")
for idx, exp in enumerate(experiments):
    print(f"{idx + 1}: Seed {exp['seed']} - Best Valid: {exp['best_valid']}%, Best Test: {exp['best_test']}%")

# Prompt user to choose an experiment to visualize
choice = int(input("Enter the number of the experiment you want to visualize: ")) - 1

# Get the chosen experiment data
chosen_experiment = experiments[choice]

# Plot Validation and Test Accuracies in one graph
plt.figure(figsize=(10, 5))
plt.plot(chosen_experiment["epochs"], chosen_experiment["valid_accuracies"], label="Validation Accuracy", marker='o', color='green')
plt.plot(chosen_experiment["epochs"], chosen_experiment["test_accuracies"], label="Test Accuracy", marker='o', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title(f"{chosen_experiment['model_type']} Model on {chosen_experiment['dataset']} - Seed {chosen_experiment['seed']} - Validation and Test Accuracy per Epoch")
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss in a separate graph
plt.figure(figsize=(10, 5))
plt.plot(chosen_experiment["epochs"], chosen_experiment["losses"], label="Loss", marker='o', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{chosen_experiment['model_type']} Model on {chosen_experiment['dataset']} - Seed {chosen_experiment['seed']} - Loss per Epoch")
plt.legend()
plt.grid(True)
plt.show()
