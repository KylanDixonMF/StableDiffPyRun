# Locally Run Stable Diffusion Model
## Installations
Install depedancies and libraries using commands
- pip install torch
- pip install transformers
- pip install Image

## Initial Load Phase
The program needs to initally download and saftensors and load files. There for an intial install run to be ran within the directory with "python Main.py"

## Potential Issues and Current Solutions
During intital load phase, if you do not have a 'cuda' compatible GPU or if you do not have the the proper software to run:

- NVIDIA GPU without cuda software: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
- FOR NON-COMPATIBLE GPU: adjust line: - pipe.to("cuda")
TO +pipe.enable_model_cpu_offload()