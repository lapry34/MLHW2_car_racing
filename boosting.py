import os
import random
import shutil

def count_files(directory):
    """Counts the number of files in a directory."""
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

def load_files(directory):
    """Loads file paths from a directory."""
    return [os.path.join(directory, name) for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))]

def copy_existing_data(src_dir, dest_dir):
    """
    Copies the existing dataset from the source directory to the destination directory.
    
    Args:
        src_dir (str): Path to the source directory.
        dest_dir (str): Path to the destination directory.
    """
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)  # Remove existing directory
    shutil.copytree(src_dir, dest_dir)  # Copy all data

def perform_boosting(directory, target_samples):
    """
    Performs boosting by sampling files to reach the target number of samples per class.
    
    Args:
        directory (str): Path to the directory containing class folders.
        target_samples (int): The number of samples to ensure for each class.
    """
    for class_label in range(5):  # Assuming 5 classes (0-4)
        class_dir = os.path.join(directory, str(class_label))
        files = load_files(class_dir)
        current_count = len(files)
        
        if current_count < target_samples:
            # Calculate how many additional samples are needed
            additional_samples = target_samples - current_count
            
            # Randomly sample files to oversample the class
            new_files = random.choices(files, k=additional_samples)
            
            # Copy the additional files to the class directory
            for idx, file_path in enumerate(new_files):
                # Create a unique name for the oversampled file
                new_file_name = f"{os.path.basename(file_path).split('.')[0]}_boosted_{idx}.jpg"
                shutil.copy(file_path, os.path.join(class_dir, new_file_name))

if __name__ == "__main__":
    target_samples = 2000  # Adjust this to the desired number of samples per class
    train_dir = 'dataset/train'
    output_dir = 'dataset/train_boosted'
    
    print(f"Copying existing dataset from '{train_dir}' to '{output_dir}'...")
    copy_existing_data(train_dir, output_dir)
    
    print("Performing boosting on the train dataset...")
    perform_boosting(output_dir, target_samples)
    
    # Display statistics
    for class_label in range(5):
        class_dir = os.path.join(output_dir, str(class_label))
        num_files = count_files(class_dir)
        print(f"Class {class_label}: {num_files} files")