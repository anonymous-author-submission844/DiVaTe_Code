from openai import OpenAI

client = OpenAI(api_key="<your_api_key>")
from PIL import Image
import requests
from io import BytesIO

def generate_image(prompt: str) -> Image.Image:

    response = client.images.generate(prompt=prompt,
    model="dall-e-3",  # DALL·E 3 via GPT-4o endpoint
    n=1,
    size="1024x1024",
    response_format="url")

    image_url = response.data[0].url
    image_data = requests.get(image_url).content
    return Image.open(BytesIO(image_data))
