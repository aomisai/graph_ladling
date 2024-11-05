import re
import matplotlib.pyplot as plt

# Initialize lists to store data
epochs = []
test_accuracies = []
valid_accuracies = []
losses = []

# Read from the file
with open("experiment_log.txt", "r") as file:
    lines = file.readlines()

# Define a pattern to match the epoch log lines with relevant data
pattern = re.compile(r"Epoch: (\d+), Loss: ([\d.]+), Train: [\d.]+%, Valid: ([\d.]+)% Test: ([\d.]+)%")
eval_step = 10  # Adjust as needed

# Parse the file content
for line in lines:
    # Only process lines that contain "Epoch" and match the pattern
    if "Epoch" in line:
        match = pattern.search(line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            valid_accuracy = float(match.group(3))
            test_accuracy = float(match.group(4))

            # Append data every eval_step epochs
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
plt.title("Validation and Test Accuracy per Epoch")
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss in a separate graph
plt.figure(figsize=(10, 5))
plt.plot(epochs, losses, label="Loss", marker='o', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()
plt.grid(True)
plt.show()
