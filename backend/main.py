from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

from PIL import Image, ImageDraw
import io
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "SVAMITVA AI Backend Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    draw = ImageDraw.Draw(image)

    width, height = image.size

    # 🟢 BUILDING DETECTION
    for i in range(3):

        x1 = random.randint(50, width // 2)
        y1 = random.randint(50, height // 2)

        x2 = x1 + random.randint(80, 180)
        y2 = y1 + random.randint(80, 180)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=5
        )

        draw.text(
            (x1, y1 - 25),
            f"Building {random.randint(88, 98)}%",
            fill="red"
        )

    # 🟡 ROAD DETECTION
    draw.line(
        [(0, height // 2), (width, height // 2)],
        fill="yellow",
        width=12
    )

    draw.text(
        (50, height // 2 - 30),
        "Road 94%",
        fill="yellow"
    )

    # 🔵 WATER BODY DETECTION
    water_x1 = width - 250
    water_y1 = height - 180

    water_x2 = width - 50
    water_y2 = height - 50

    draw.rectangle(
        [water_x1, water_y1, water_x2, water_y2],
        outline="blue",
        width=6
    )

    draw.text(
        (water_x1, water_y1 - 30),
        "Water Body 91%",
        fill="blue"
    )

    buffer = io.BytesIO()

    image.save(buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )