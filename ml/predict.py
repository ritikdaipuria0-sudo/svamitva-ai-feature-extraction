import torch
import cv2
import numpy as np
from model import SimpleUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleUNet().to(device)
model.load_state_dict(torch.load("../saved_models/model.pth", map_location=device))
model.eval()

image_path = "../data/images/image_0.png"

image = cv2.imread(image_path)
image = cv2.resize(image, (128, 128))
input_image = image / 255.0

input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

colored_mask = np.zeros((128, 128, 3), dtype=np.uint8)

colored_mask[prediction == 1] = [0, 255, 0]      # Building = Green
colored_mask[prediction == 2] = [255, 255, 0]    # Road = Yellow
colored_mask[prediction == 3] = [255, 0, 0]      # Water = Blue

overlay = cv2.addWeighted(image, 0.6, colored_mask, 0.4, 0)

cv2.imwrite("../data/prediction_output.png", overlay)

print("Prediction completed!")
print("Output saved at: data/prediction_output.png")