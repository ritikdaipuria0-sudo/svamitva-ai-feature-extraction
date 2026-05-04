import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model import SimpleUNet
from dataset import DroneDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DroneDataset("../data/images", "../data/masks")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = SimpleUNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "../saved_models/model.pth")
print("Model trained and saved!")