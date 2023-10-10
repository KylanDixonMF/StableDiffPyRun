# Locally Run Stable Diffusion Model
## Installations
Install depedancies and libraries using commands
- pip install torch
- pip install transformers
- pip install Image
- pip install accelerate safetensors

## Initial Load Phase
The program needs to initally download and saftensors and load files. There for an intial install run to be ran within the directory with "python Main.py" 

## Version to Run
Base.py will work on windows. To run CPU instead of GPU if not using CUDA 12.1 and pytorch run command: pip3 install torch torchvision torchaudio

## Potential Issues and Current Solutions
During intital load phase, if you do not have a 'cuda' compatible GPU or if you do not have the the proper software to run:

- NVIDIA GPU without CUDA drivers version 12.1 is compatible with pytorch: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local  
- For windows: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
- For linux: pip3 install torch torchvision torchaudio

- FOR NON-COMPATIBLE GPU: adjust line: - pipe.to("cuda")
TO +pipe.enable_model_cpu_offload()
LINUX: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
WINDOWS: pip3 install torch torchvision torchaudio

## Current error to fix: Traceback (most recent call last):
  File "/home/autonomyllc/Desktop/SDXL/StableDiffPyRun/Main.py", line 44, in <module>
    img = transforms.ToPILImage()(image)
  File "/home/autonomyllc/.local/lib/python3.10/site-packages/torchvision/transforms/transforms.py", line 234, in __call__
    return F.to_pil_image(pic, self.mode)
  File "/home/autonomyllc/.local/lib/python3.10/site-packages/torchvision/transforms/functional.py", line 262, in to_pil_image
    raise TypeError(f"pic should be Tensor or ndarray. Got {type(pic)}.")
TypeError: pic should be Tensor or ndarray. Got <class 'PIL.Image.Image'>.