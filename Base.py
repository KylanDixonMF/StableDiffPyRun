from diffusers import DiffusionPipeline
import torch
import os

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.pipe.enable_model_cpu_offload()

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = input("Enter prompt: ")

images = pipe(prompt=prompt).images[0]

# Specify the directory where you want to save the image
save_directory = "/home/autonomyllc/Desktop/SDXL/output"  # Replace with your specific directory path

# Define the filename for the saved image
filename = "generated_image.png"  # You can change the filename as needed

# Construct the full path for saving the image
save_path = os.path.join(save_directory, filename)

# Save the image
images.save(save_path)

# Print the path to the saved image
print(f"Image saved at: {save_path}")