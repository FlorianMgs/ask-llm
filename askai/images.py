import base64
import requests
from PIL import Image
from io import BytesIO

import os


def is_base64(sb):
    try:
        if isinstance(sb, str):
            base64.b64decode(sb)
            return True
        else:
            return False
    except Exception:
        return False


def is_url(s):
    return s.startswith("http://") or s.startswith("https://")


def image_to_b64(image: Image.Image, image_format: str | None = None) -> str:
    image_io = BytesIO()
    image_format = image_format.upper() if image_format else "WEBP"
    try:
        image.save(image_io, image_format)
    except KeyError:
        image.save(image_io, format="WEBP")
    encoded = base64.b64encode(image_io.getvalue()).decode()
    return f"data:image/{image_format.lower()};base64,{encoded}"


def download_image_as_pil_image(url: str) -> Image.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def convert_image_to_base64(input_image: str | Image.Image) -> str:
    if isinstance(input_image, Image.Image):
        return image_to_b64(input_image)
    elif isinstance(input_image, str):
        if is_base64(input_image):
            return input_image
        elif is_url(input_image):
            pil_image = download_image_as_pil_image(input_image)
            b64 = image_to_b64(pil_image)
            return b64
        elif os.path.isfile(input_image):
            pil_image = Image.open(input_image)
            return image_to_b64(pil_image)
        else:
            raise ValueError(
                "Invalid input: Input string is not a valid path, URL, or base64 string."
            )
    else:
        raise TypeError(
            "Invalid input type: Input must be a PIL Image, a file path, a URL, or a base64 string."
        )
