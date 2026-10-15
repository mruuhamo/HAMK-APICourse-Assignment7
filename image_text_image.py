"""Exercise #7: Image -> Text -> Image.

Reads an image, asks a vision model (gpt-5.6-luna) for a detailed textual
description, prints it, then uses that description as the prompt for an
image-generation model (gpt-image-2) and saves the result locally.

This is deliberately NOT image-to-image: the generator never sees the
original image, only the vision model's textual interpretation of it.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import mimetypes
import os
from datetime import datetime
from pathlib import Path

import openai
from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_VISION_MODEL = "gpt-5.6-luna"
DEFAULT_IMAGE_MODEL = "gpt-image-2"
DEFAULT_OUTPUT_DIR = Path("output")

SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
SIZES = ["auto", "1024x1024", "1536x1024", "1024x1536"]
QUALITIES = ["auto", "low", "medium", "high"]

DESCRIPTION_PROMPT = """\
Describe this image in rich detail so that an image-generation model could
recreate it as faithfully as possible from your text alone.

Cover:
- main subject(s): appearance, pose, clothing, expression, count and placement
- setting and background
- composition and framing, camera angle, perspective, focal length / depth of field
- colors and color palette
- lighting: direction, quality, time of day, shadows
- medium and style (e.g. photograph, oil painting, 3D render, flat illustration)
- textures and materials
- mood and atmosphere
- any visible text, reproduced exactly in quotes

Write the result as a single, flowing image-generation prompt in English
(one or a few paragraphs). Do not add commentary, headings or bullet points,
and do not mention that you are describing an image.
"""


def image_to_data_url(filename: str | Path) -> str:
    """Read a local image and return it as a Base64 data URL."""
    path = Path(filename)

    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in SUPPORTED_MIME_TYPES:
        raise ValueError(
            f"Unsupported image type: {mime_type}. "
            f"Supported types: {sorted(SUPPORTED_MIME_TYPES)}"
        )

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def describe_image(client: OpenAI, image_path: Path, model: str) -> str:
    """Send the image to a vision-capable model and return its description."""
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": DESCRIPTION_PROMPT},
                    {"type": "input_image", "image_url": image_to_data_url(image_path)},
                ],
            }
        ],
    )

    description = (response.output_text or "").strip()
    if not description:
        raise RuntimeError("The vision model returned an empty description.")
    return description


def generate_image(
    client: OpenAI,
    prompt: str,
    model: str,
    size: str,
    quality: str,
) -> bytes:
    """Generate an image from a text prompt and return the raw image bytes."""
    result = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        n=1,
    )

    if not result.data or not result.data[0].b64_json:
        raise RuntimeError("The image API did not return image data.")

    try:
        return base64.b64decode(result.data[0].b64_json, validate=True)
    except binascii.Error as exc:
        raise RuntimeError("Invalid Base64 image data received.") from exc


def save_bytes(data: bytes, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path.resolve()


def default_output_path(image_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"{image_path.stem}_regenerated_{timestamp}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Image -> Text -> Image: describe an image with a vision model, "
            "then generate a new image from that description."
        ),
    )
    parser.add_argument("image", type=Path, help="input image (.jpg, .jpeg, .png, .webp)")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="output PNG path (default: output/<name>_regenerated_<timestamp>.png)",
    )
    parser.add_argument("--size", choices=SIZES, default="auto", help="generated image size (default: auto)")
    parser.add_argument("--quality", choices=QUALITIES, default="medium", help="generated image quality (default: medium)")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL, help=f"vision model (default: {DEFAULT_VISION_MODEL})")
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL, help=f"image model (default: {DEFAULT_IMAGE_MODEL})")
    parser.add_argument(
        "--save-description", action="store_true",
        help="also save the description as a .txt file next to the generated image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file or your environment."
        )

    # Validate the input before making any API calls.
    image_to_data_url(args.image)
    output_path = args.output or default_output_path(args.image)

    client = OpenAI()

    print(f"[1/3] Describing '{args.image}' with {args.vision_model} ...")
    description = describe_image(client, args.image, args.vision_model)

    print("\n=== Description ===")
    print(description)
    print("===================\n")

    print(f"[2/3] Generating image with {args.image_model} (size={args.size}, quality={args.quality}) ...")
    image_bytes = generate_image(client, description, args.image_model, args.size, args.quality)

    saved = save_bytes(image_bytes, output_path)
    print(f"[3/3] Saved image: {saved}")

    if args.save_description:
        text_path = save_bytes(description.encode("utf-8"), output_path.with_suffix(".txt"))
        print(f"      Saved description: {text_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, openai.OpenAIError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
