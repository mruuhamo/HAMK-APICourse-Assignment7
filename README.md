# Exercise #7: Image → Text → Image

A command-line app that:

1. reads an image
2. sends it to a vision-capable model (`gpt-5.6-luna`, OpenAI Responses API)
3. generates a detailed textual description
4. prints the description
5. uses that description as the prompt for an image-generation model (`gpt-image-2`, OpenAI Image API)
6. saves the generated image locally

```
input.jpg ──► gpt-5.6-luna ──► text description ──► gpt-image-2 ──► output/*.png
```

## This is not image-to-image

The image generator never sees the original image. It only receives the vision
model's text. The result is therefore a *reinterpretation* of the original:
composition, style and subject usually survive, but exact faces, small details,
exact counts, positions and text often change. Comparing the input and output
shows what the vision model noticed and what got lost in the text step.

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then put your OpenAI API key in .env
```

You can also set `OPENAI_API_KEY` as a normal environment variable instead of using `.env`.

## Usage

```powershell
python image_text_image.py photo.jpg
python image_text_image.py photo.jpg -o output\sauna.png --size 1536x1024 --quality high
python image_text_image.py photo.png --save-description
```

| Option | Default | Description |
|---|---|---|
| `image` | (required) | Input image: `.jpg`, `.jpeg`, `.png` or `.webp` |
| `-o`, `--output` | `output/<name>_regenerated_<timestamp>.png` | Where to save the generated image |
| `--size` | `auto` | `auto`, `1024x1024`, `1536x1024`, `1024x1536` |
| `--quality` | `medium` | `auto`, `low`, `medium`, `high` |
| `--vision-model` | `gpt-5.6-luna` | Vision model for the description |
| `--image-model` | `gpt-image-2` | Image-generation model |
| `--save-description` | off | Also saves the description as a `.txt` file next to the image |

Example output:

```
[1/3] Describing 'photo.jpg' with gpt-5.6-luna ...

=== Description ===
A photorealistic wide-angle photograph of a small red wooden sauna on the
shore of a calm Finnish lake at sunset ...
===================

[2/3] Generating image with gpt-image-2 (size=auto, quality=medium) ...
[3/3] Saved image: C:\...\Assignment7\output\photo_regenerated_20261015-151200.png
```

## How it works

- `image_to_data_url()` checks that the file exists and is a supported type, then
  encodes it as a Base64 data URL (same approach as the course material).
- `describe_image()` calls `client.responses.create()` with an `input_text`
  prompt and an `input_image`. The prompt asks for a detailed description
  (subject, composition, camera, colors, lighting, style, textures, mood,
  visible text) written as a ready-to-use image-generation prompt.
- `generate_image()` calls `client.images.generate()` and decodes the returned
  `b64_json` with validation.
- Errors (missing file, unsupported type, missing API key, API errors) are
  printed as `Error: ...` and the program exits with a non-zero status.

## Cost note

Each run makes one vision call and one `gpt-image-2` generation. Use
`--quality low` while experimenting to keep costs down.
