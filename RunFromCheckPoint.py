from diffusers import DiffusionPipeline
import torch
from PIL import Image
import torchvision.transforms as transforms

print(torch.version.cuda)
print(torch.cuda.is_available())

# Load your base model checkpoint
base_checkpoint = torch.load("/home/autonomyllc/Desktop/SDXL/outputsd-pokemon-model/checkpoint-3000")

# Modify the following line to load your base checkpoint
base = DiffusionPipeline.from_pretrained(
    base_checkpoint,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

base.to("cuda")

# Define how many steps and what % of steps to be run on each expert (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = input("\nEnter image prompt: \n")

# Run the base model
image = base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images

# Assuming image is a tensor
image.save("/home/autonomyllc/Desktop/SDXL/output/genImage.png")
