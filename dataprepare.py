import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_metadata(root_dir):
    """Generate structured metadata for the dataset."""
    data = []
    try:
        # List directories (each representing a class)
        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for class_name in classes:
            class_path = os.path.join(root_dir, class_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in images:
                data.append({
                    "path": os.path.join(class_path, img_file),
                    "label": class_name
                })
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    # Update dataset_path if needed
    dataset_path = r"Project/Medicinal plant dataset"
    df = create_metadata(dataset_path)
    
    if df is not None:
        df.to_csv("metadata.csv", index=False)
        print(f"Metadata created with {len(df)} samples")
        print(f"Unique classes: {df['label'].nunique()}")
        
        # Create stratified train-val split
        train_df, val_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df["label"], 
            random_state=42
        )
        train_df.to_csv("train.csv", index=False)
        val_df.to_csv("val.csv", index=False)
        print(f"\nTraining samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
