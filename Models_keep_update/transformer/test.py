import torch, torchvision, torchaudio, torchtext
print(torch.__version__)       # 2.5.1+cu121
print(torchvision.__version__) # 0.20.1+cu121
print(torchtext.__version__)   # 0.18.0
print(torch.cuda.is_available())  # True(如果驱动OK)