import torch
print(torch.version.cuda)         # Should not be None
print(torch.backends.cudnn.version())  # Should give a version if CUDA is available
print(torch.cuda.is_available()) # Should be True if working
