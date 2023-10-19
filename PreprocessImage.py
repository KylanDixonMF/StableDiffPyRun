import tensorflow as tf

# Define the image preprocessing function
def load_and_preprocess_image(image_path, target_size=(512, 512)): # change dims as needed
    # Load the image from the file path
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)  # Assuming RGB images
    
    image = tf.image.resize(image, target_size, preserve_aspect_ratio=True)
    
    # Resize the image to the target size (e.g., 1024x1024) this is the base size for sdxl
    image = tf.image.resize(image, target_size)
    
    # Normalize the pixel values to the range [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # You can apply additional preprocessing steps here if needed
    
    return image

