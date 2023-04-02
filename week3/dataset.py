from pathlib import Path
from typing import List, Dict

from PIL import Image
import os


def create_detectron_dataset(dataset_path: str) -> List[Dict]:
    """
    Creates a list of dictionaries in Detectron format from a directory of image files in COCO FORMAT.
    Args:
        dataset_path (Path): Path to the directory where the images are stored.
    Returns:
        List[Dict]: A list of dictionaries in Detectron format.
    """
    dataset = []
    # given the path to the directory, get all the image files
    image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if
                   f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

    for i, img_file in enumerate(image_files):
        img = Image.open(img_file)

        dataset.append({
            "file_name": str(img_file),
            "height": img.height,
            "width": img.width,
            "image_id": i,
        })

    return dataset
