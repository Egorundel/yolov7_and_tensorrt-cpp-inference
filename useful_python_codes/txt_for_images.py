import os


# A function that runs through images and saves paths to a file that you specify below in the output_file variable
def save_file_paths(root_dir, file):
    with open(file, 'w') as f:
        for root, dirs, files in os.walk(root_dir):
            for filename in files:
                file_path = './' + os.path.join(root, filename)[os.path.join(root, filename).find("images") + 0:]
                print(file_path)
                f.write(file_path + '\n')


# Path to folder images/train (images/val, images/test)
train_dir = 'images/train'

# Path to file, in which will be saved paths to pictures
output_file = 'train.txt'

# Starting file crawling and saving image paths to output_file
save_file_paths(train_dir, output_file)