import tensorflow as tf
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from transformers import DiffusionModel
from PreprocessImage import load_and_preprocess_image 

# Define the path to your CSV and Parquet files
csv_file_path = "your_dataset.csv"  # CSV file path
parquet_file_path = "your_dataset.parquet"  # Parquet file path

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
batch_size = 4
epochs = 5
learning_rate = 1e-5
weight_decay = 1e-2

# Load the pretrained model
pretrained_model = DiffusionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")

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
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Shuffle your dataset for each epoch if needed
    dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    for start in range(0, len(dataset), batch_size):
        end = start + batch_size
        batch = dataset[start:end]
        
        # You should load and preprocess your images and captions here
        images = batch['image_path'].apply(load_and_preprocess_image).to_numpy()
        captions = batch['caption'].to_numpy()
        
        loss = diffusion_ft_trainer.train_on_batch([images, captions], images)  # Adjust inputs and targets as needed
        print(f"Batch loss: {loss}")

# Save the fine-tuned model
diffusion_ft_trainer.save("fine_tuned_diffusion_model.h5") #edit path for model
