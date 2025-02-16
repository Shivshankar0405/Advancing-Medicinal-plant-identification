import os
import pandas as pd

class DataHandler:
    @staticmethod
    def validate_files():
        """Check for required data files."""
        required = ["metadata.csv", "train.csv", "val.csv"]
        missing = [f for f in required if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"Missing files: {missing}. Run dataprepare.py first")
            
    @staticmethod
    def load_metadata():
        """Load all metadata files."""
        DataHandler.validate_files()
        return (
            pd.read_csv("train.csv"),
            pd.read_csv("val.csv"),
            pd.read_csv("metadata.csv")
        )

if __name__ == "__main__":
    try:
        train_df, val_df, metadata_df = DataHandler.load_metadata()
        print("Data validation successful!")
        print(f"Total classes: {metadata_df['label'].nunique()}")
    except Exception as e:
        print(f"Error: {str(e)}")
