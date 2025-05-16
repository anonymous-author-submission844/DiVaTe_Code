from PIL import Image
import requests
from io import BytesIO

def generate_image(prompt: str) -> Image.Image:
    # Replace with real Gemini API
    response = requests.get("https://via.placeholder.com/512.png?text=Gemini")
    return Image.open(BytesIO(response.content))
