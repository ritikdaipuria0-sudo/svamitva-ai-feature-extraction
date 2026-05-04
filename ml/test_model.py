import torch
from model import SimpleUNet

model = SimpleUNet()

dummy_image = torch.randn(1, 3, 128, 128)

output = model(dummy_image)

print("Input shape:", dummy_image.shape)
print("Output shape:", output.shape)
print("Model working successfully!")