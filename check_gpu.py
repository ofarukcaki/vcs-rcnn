import torch

print("CUDA is available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA device available. Using:", torch.device('cpu'))
