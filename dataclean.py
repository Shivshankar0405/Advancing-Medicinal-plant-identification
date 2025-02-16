import os
from PIL import Image
from tqdm import tqdm

def check_corrupted_images(root_dir):
    """Detect and remove corrupted images."""
    corrupted = []
    try:
        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for class_dir in tqdm(classes, desc="Checking classes"):
            class_path = os.path.join(root_dir, class_dir)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in tqdm(images, desc=f"Checking {class_dir}"):
                img_path = os.path.join(class_path, img_file)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    corrupted.append(img_path)
    except Exception as e:
        print(f"Error: {str(e)}")
    return corrupted

if __name__ == "__main__":
    dataset_path = r"C:/Users/shrut/Desktop/project/Medicinal plant dataset"
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset directory not found at {dataset_path}")
        exit(1)
        
    corrupted = check_corrupted_images(dataset_path)
    print(f"\nFound {len(corrupted)} corrupted images.")
    if corrupted:
        for img in corrupted:
            os.remove(img)
        print("Deleted corrupted images.")