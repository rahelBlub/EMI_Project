import os
from PIL import Image
import io


def bytes_to_pil(b):
    """JPEG-Bytes -> PIL-Image"""
    return Image.open(io.BytesIO(b))

def get_all_single_image_paths(image_directory):
    skipped = 0
    for image in os.listdir(image_directory):
        image_path = os.path.join(image_directory, image)
        if not os.path.exists(image_path):
            skipped += 1
            continue

    if skipped:
        print(f"{skipped} Eintraege uebersprungen (Bild fehlt oder Fehler).")