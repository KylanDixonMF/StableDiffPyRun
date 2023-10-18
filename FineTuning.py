import tensorflow as tf
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from diffusers import DiffusionPipeline
from PreprocessImage import load_and_preprocess_image 

gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
if gpus:
    for gpu in gpus:
        print("GPU:", gpu.name)

# Define the path to your CSV and Parquet files
csv_file_path = "your_dataset.csv"  # CSV file path
parquet_file_path = "/home/autonomyllc/Desktop/SDXL/StableDiffPyRun/ExampleDataset/train-00000-of-00001-566cc9b19d7203f8.parquet"  # Parquet file path

# Define a function to load data from CSV
def load_data_from_csv(csv_file_path):
    return pd.read_csv(csv_file_path)

# Define a function to load data from Parquet
def load_data_from_parquet(parquet_file_path):
    parquet_file = pq.read_table(parquet_file_path)
    return parquet_file.to_pandas()

# Choose the dataset format here (CSV or Parquet)
use_csv = False  # Set this to True for CSV, False for Parquet

# Load your custom dataset based on the chosen format
if use_csv:
    dataset = load_data_from_csv(csv_file_path)
else:
    dataset = load_data_from_parquet(parquet_file_path)

# Define hyperparameters
batch_size = 24
epochs = 5
learning_rate = 1e-4
weight_decay = 1e-2

# Load the pretrained model
pretrained_model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

# Set up the Trainer
diffusion_ft_trainer = tf.keras.Sequential([
    pretrained_model,  # Use the pretrained model as the first layer
    # Add any additional layers or operations for your fine-tuning task here
])

# Define the loss function and optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, weight_decay=weight_decay)

# Compile the model
diffusion_ft_trainer.compile(optimizer=optimizer, loss=loss_fn)

# Fine-tune the model
# Replace this with your dataset and appropriate preprocessing
# Make sure your dataset contains columns 'image_path' and 'caption'
image_column_name = 'image'  # Adjust to your column name
caption_column_name = 'text'  # Adjust to your column name

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Shuffle your dataset for each epoch if needed
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    for start in range(0, len(dataset), batch_size):
        end = start + batch_size
        batch = dataset[start:end]
        
        # You should load and preprocess your images and captions here
        images = batch[image_column_name].apply(load_and_preprocess_image).to_numpy()
        captions = batch[caption_column_name].to_numpy()
        
        print(f"Batch start: {start}, end: {end}")
        print(f"Number of images in batch: {len(images)}")
        print(f"Number of captions in batch: {len(captions)}")
        
        loss = diffusion_ft_trainer.train_on_batch([images, captions], images)  # Adjust inputs and targets as needed
        print(f"Batch loss: {loss}")

# Save the fine-tuned model
diffusion_ft_trainer.save("/home/autonomyllc/Desktop/SDXL/fine_tuned_diffusion_model.h5") #edit path for model
