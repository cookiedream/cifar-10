import torch

if torch.cuda.is_available():
    device = torch.cuda.get_device_name()
    print("GPU CUDA is available.")
    print("GPU Model:", device)
else:
    print("GPU CUDA is not available.")
