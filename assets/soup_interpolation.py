import os
import torch
import numpy as np
from collections import defaultdict
from options.base_options import BaseOptions
from trainer import trainer



# Function to perform interpolation between two models and find the optimal mix
def interpolate(state1, state2, model, data, split_idx, evaluator, trainer_instance, granularity=10):
    """
    Interpolates between two state dictionaries (state1 and state2) and returns
    the best interpolation factor that maximizes validation accuracy.
    """
    alpha = np.linspace(0, 1, granularity)  # Granularity controls the number of interpolation steps
    max_val, loc = -1, -1
    best_sd = None

    for i in alpha:
        sd = {}
        for k in state1.keys():
            sd[k] = state1[k].clone() * i + state2[k].clone() * (1 - i)
        model.load_state_dict(sd)
        valid_acc = test_single_state_dict(trainer_instance, sd)

        if valid_acc > max_val:
            max_val = valid_acc
            loc = i  # Save the best interpolation factor
            best_sd = sd  # Save the best state_dict

    return max_val, loc, best_sd  # Return the best validation accuracy, the interpolation factor, and the final state_dict


def test_single_state_dict(trainer_instance, state_dict):
    """
    Function to test a single state_dict using the trainer_instance and return its validation accuracy.

    Parameters:
    - trainer_instance: An instance of the 'trainer' class (initialized with args, data, and model).
    - state_dict: A single state dictionary (loaded model weights).

    Returns:
    - valid_acc: The validation accuracy of the model with the given state_dict.
    """
    # Load the state dictionary into the model
    trainer_instance.model.load_state_dict(state_dict)

    # Test the model using the test_net function from the trainer
    _, (train_acc, valid_acc, test_acc) = trainer_instance.test_net()

    # Return the validation accuracy
    return valid_acc


def test_state_dicts(trainer_instance, state_dicts):
    """
    Function to test a list of state_dicts and return them ordered by validation accuracy.

    Parameters:
    - trainer_instance: An instance of the 'trainer' class (initialized with args, data, and model).
    - state_dicts: A list of state dictionaries (loaded model weights).

    Returns:
    - List of state dictionaries sorted by validation accuracy (from highest to lowest).
    - List of corresponding validation accuracies (sorted from highest to lowest).
    """
    results = []

    for idx, state_dict in enumerate(state_dicts):
        print(f"Testing state_dict {idx + 1}/{len(state_dicts)}...")

        # Get the validation accuracy for the current state_dict
        valid_acc = test_single_state_dict(trainer_instance, state_dict)

        # Append the validation accuracy and corresponding state_dict to the results
        results.append((valid_acc, state_dict))

        # Print validation accuracy
        print(f"Validation Accuracy for state_dict {idx + 1}: {valid_acc * 100:.2f}%")

    # Sort results by validation accuracy (from highest to lowest)
    results_sorted = sorted(results, key=lambda x: x[0], reverse=True)

    # Extract the sorted state_dicts and validation accuracies
    sorted_state_dicts = [state_dict for _, state_dict in results_sorted]
    sorted_validation_accuracies = [valid_acc for valid_acc, _ in results_sorted]

    return sorted_state_dicts


    #
    # # Perform the testing process using the trainer's test_net() method
    # _, (train_acc, valid_acc, test_acc) = trainer.test_net()
    #
    # print(f"Train Accuracy: {train_acc * 100:.2f}%")
    # print(f"Validation Accuracy: {valid_acc * 100:.2f}%")
    # print(f"Test Accuracy: {test_acc * 100:.2f}%")
    #
    # return train_acc, valid_acc, test_acc


# Function to load model state_dicts from files
def load_model_files(model_files, model_dir):
    state_dicts = []
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)  # Get full path to the model file
        state_dict = torch.load(model_path)  # Load model weights
        state_dicts.append(state_dict)
    return state_dicts


# Function to perform the greedy interpolation soup procedure
def greedy_interpolation_soup(state_dicts, model, data, split_idx, evaluator, trainer):
    """
    Greedy Interpolation Soup procedure as described in the paper.

    Args:
    - state_dicts: List of state dictionaries (model weights).
    - model: The model architecture.
    - data, split_idx, evaluator: Data and evaluator for the model.

    Returns:
    - final_soup: The final souped state dictionary after greedy interpolation.
    """
    print("Sorting models by their validation accuracy...")

    # Step 1: Sort the models by their validation accuracy
    state_dicts_sorted = test_state_dicts(trainer, state_dicts)

    # Step 2: Start the soup with the model that has the highest validation accuracy
    soup = state_dicts_sorted[0]
    print(f"Starting soup with the best model (highest validation accuracy).")

    # Step 3: For each model, try to interpolate with the current soup
    for i in range(1, len(state_dicts_sorted)):
        current_model = state_dicts_sorted[i]
        print(f"Trying to interpolate model {i} into the soup...")

        # Save current soup validation accuracy for comparison
        current_soup_val_acc = test_single_state_dict(trainer, soup)

        # Interpolate between the current soup and the new model
        best_val_acc, best_alpha, best_sd = interpolate(soup, current_model, model, data, split_idx, evaluator, trainer)

        # If the validation accuracy improves compared to the current soup
        if best_val_acc > current_soup_val_acc:
            print(f"Model {i} improved the validation accuracy, adding to soup with alpha = {best_alpha}")
            soup = best_sd  # Update soup with the best interpolated state
        else:
            print(f"Model {i} did not improve the validation accuracy, discarding.")

    return soup  # Return the final souped model


# Function to group model files by dataset and model type
def group_model_files(model_files):
    grouped_files = defaultdict(list)

    for file in model_files:
        # Extract model type and dataset from the file name (e.g., "model_SGC_seed_0_dataset_Flickr_lr_0.001_wd_0.0.pth")
        parts = file.split('_')
        model_type = parts[1]  # e.g., "SGC"
        dataset = parts[5]     # e.g., "Flickr"
        key = (dataset, model_type)  # Group by dataset and model type
        grouped_files[key].append(file)

    return grouped_files


def main():
    # Step 1: Get model files (assuming they are in the "trained_soup_ingredients" directory)
    model_dir = "trained_soup_ingredients"
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') and f.startswith('model')]

    if len(model_files) < 2:
        print("Need at least two model files to perform interpolation!")
        return

    # Group the model files by dataset and model type
    grouped_files = group_model_files(model_files)

    # If multiple groups are found, present them to the user
    print("Possible interpolation soup procedures:")
    for idx, (key, files) in enumerate(grouped_files.items(), start=1):
        dataset, model_type = key
        print(f"{idx}. Dataset: {dataset}, Model: {model_type}, {len(files)} files total")

    # Ask the user to choose a group
    choice = int(input("Enter the number corresponding to the group you want to perform interpolation on: ")) - 1

    # Get the corresponding group of files based on user choice
    selected_group = list(grouped_files.items())[choice][1]

    # Load model state_dicts from the selected group of files
    state_dicts = load_model_files(selected_group, model_dir)

    # Step 2: Initialize the args and trainer, load model and data just like main.py
    args = BaseOptions().initialize()  # Load the arguments like in main.py

    # Check CUDA availability and set device
    if torch.cuda.is_available():
        args.cuda = True
        args.device = torch.device(f"cuda:{args.cuda_num}")
        print("CUDA available")
    else:
        print("CUDA is not available. Switching to CPU.")
        args.cuda = False
        args.device = torch.device("cpu")

    trainer_instance = trainer(args)  # Initialize the trainer to handle model and data loading

    # Now use the trainer's model and data
    model = trainer_instance.model  # This is the actual model used in the training/testing
    data = trainer_instance.data  # This is the loaded dataset
    split_idx = trainer_instance.split_masks  # These are the train/validation/test splits
    evaluator = trainer_instance.evaluator  # The evaluator object, if applicable

    # Step 3: Perform the greedy interpolation soup procedure
    print(f"Performing greedy interpolation soup procedure on {len(selected_group)} files...")
    final_soup = greedy_interpolation_soup(state_dicts, model, data, split_idx, evaluator, trainer_instance)

    # Step 4: Save the final souped model with the filename including model type and dataset
    model_type = trainer_instance.args.type_model  # Get model type (e.g., 'SGC')
    dataset = trainer_instance.args.dataset  # Get dataset name (e.g., 'Flickr')

    # Construct the filename with the model type and dataset
    souped_model_filename = f"final_soup_model_{model_type}_{dataset}.pth"

    # Create the "completed_soups" directory if it doesn't exist
    soups_dir = "completed_soups"
    if not os.path.exists(soups_dir):
        os.makedirs(soups_dir)

    # Modify the save path to use the "completed_soups" directory
    souped_model_filename = os.path.join(soups_dir, souped_model_filename)
    torch.save(final_soup, souped_model_filename)
    print(f"Final souped model saved as '{souped_model_filename}'")

    soup_val_acc = test_single_state_dict(trainer_instance, final_soup)
    print(f"The validation accuracy for the soups is {soup_val_acc * 100:.2f}")


if __name__ == "__main__":
    main()