import os

def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

if __name__ == "__main__":
    for type in ("train", "test"):
        print(f"Folder {type}: ")
        total_files = sum(count_files(f'dataset/{type}/{i}') for i in range(5))
        for i in range(5):
            num_files = count_files(f'dataset/{type}/{i}')
            percentage = (num_files / total_files) * 100 if total_files > 0 else 0
            print(f"{i}: {num_files} ({percentage:.2f}%)")