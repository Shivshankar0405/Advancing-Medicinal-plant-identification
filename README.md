# Advancing-Medicinal-plant-identification
Plant Identification using Deep Learning: A full pipeline that cleans and augments data, generates metadata, and trains a ResNet-18 model on 70 classes. It also provides a CLI for training, evaluation, and prediction.
Plant Identification using Deep Learning
This repository contains a complete pipeline for automated plant identification using deep learning. The project leverages transfer learning with a pre-trained ResNet-18 model and integrates robust data preprocessing, augmentation, training, evaluation, and prediction functionalities. The pipeline is designed to classify plant images into 70 distinct classes.

Features
Data Cleaning:
Remove corrupted images from the dataset using dataclean.py.

Metadata Preparation:
Generate structured metadata (image paths and labels) from the dataset folder structure with dataprepare.py.

Data Loading:
Load and validate the metadata using dataload.py, ensuring the dataset is ready for training.

Data Augmentation:
Apply various augmentation techniques (e.g., resizing, random rotations, flips, brightness/contrast adjustments) using Albumentations in dataaug.py.

Model Training:
Fine-tune a pre-trained ResNet-18 to classify plant images into 70 classes. The training loop, implemented in model.py, includes gradient clipping, learning rate scheduling, and checkpoint saving.

Model Evaluation:
Evaluate model performance with detailed classification reports using evaluate.py.

Command-line Interface:
Use interface.py to easily train, evaluate, and predict from the command line.

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/plant-identification.git
cd plant-identification
Create and activate a virtual environment:

bash
Copy
Edit
python -m venv .venv
.\.venv\Scripts\activate   # on Windows
# or
source .venv/bin/activate   # on Linux/Mac
Install the required packages:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Data Preparation:
Run dataprepare.py to generate the metadata:

bash
Copy
Edit
python dataprepare.py
Training:
Train the model using the command-line interface:

bash
Copy
Edit
python interface.py train --epochs 15
Evaluation:
Evaluate the trained model:

bash
Copy
Edit
python interface.py evaluate --checkpoint best_model.pth
Prediction:
Predict the class of a new image:

bash
Copy
Edit
python interface.py predict --image "path/to/your/image.jpg" --checkpoint best_model.pth
Contributing
Contributions, suggestions, and improvements are welcome! Feel free to open issues or submit pull requests.

License
This project is licensed under the MIT License.
