import os

def count_png_files(parent_directory):
    png_count = 0
    for _, _, files in os.walk(parent_directory):
        png_count += sum(1 for file in files if file.lower().endswith('.png'))
    return png_count

def count_images_in_subdirectories(parent_directory, extensions=(".png", ".jpg", ".jpeg")):
    image_counts = {}
    
    for subdir in os.listdir(parent_directory):
        subdir_path = os.path.join(parent_directory, subdir)
        if os.path.isdir(subdir_path):
            count = sum(
                1 for file in os.listdir(subdir_path) 
                if file.lower().endswith(extensions)
            )
            image_counts[subdir] = count
    return image_counts
