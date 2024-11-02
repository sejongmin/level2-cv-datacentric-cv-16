import json
from pathlib import Path

def read_json(filename):
    with Path(filename).open(encoding='utf8') as f:
        ann = json.load(f)
    return ann

def reshape_points(x): 
    return [x[0:2], x[2:4], x[4:6], x[6:8]]

def wrap_points(points):
    return {
        "transcription": "",
        "points": points,
        "orientation": "",
        "language": None,
        "tags": [],
        "confidence": None,
        "illegibility": False,
    }

def wrap_words(words, wh):
    return {
        "paragraphs": {},
        "words": words,
        "chars": {},
        "img_w": wh[0],
        "img_h": wh[1],
        "num_patches": None,
        "tags": [],
        "relations": {},
        "annotation_log": {
            "worker": "worker",
            "timestamp": "2099-05-30",
            "tool_version": "",
            "source": None,
        },
        "license_tag": {
            "usability": True,
            "public": False,
            "commercial": True,
            "type": None,
            "holder": "Upstage",
        },
    }

def wrap_images(images): 
    return {"images": images, "version": "990530", "tags": []}

def get_image_dimensions(image_datum_aro):
    """
    Get image dimensions from datumaro annotation, with fallback to frame attributes
    """
    try:
        # Try to get from info field first
        return (image_datum_aro["info"]["img_w"], image_datum_aro["info"]["img_h"])
    except (KeyError, TypeError):
        # Fallback to frame attributes if info is missing
        try:
            return (
                image_datum_aro["frame"]["width"], 
                image_datum_aro["frame"]["height"]
            )
        except (KeyError, TypeError):
            # If no dimensions available, use default values
            # You might want to adjust these defaults or raise an error instead
            return (1024, 1024)

def make_img_bboxes_map(image_datum_aro):
    """
    Convert a single image annotation from datumaro to UFO format
    Now handles cases where info field is missing
    """
    # Extract filename from id, handling different path formats
    filename = image_datum_aro["id"].split("/")[-1]
    
    # Get image dimensions with fallback options
    dimensions = get_image_dimensions(image_datum_aro)
    
    # Create words dictionary from annotations
    words = {
        f"{idx+1:04}": wrap_points(reshape_points(annotation["points"]))
        for idx, annotation in enumerate(image_datum_aro["annotations"])
    }
    
    return (filename, wrap_words(words, dimensions))

def datum_aro_2_ufo_reduced(data):
    """
    Convert datumaro format to UFO format
    Now handles missing info field gracefully
    """
    return wrap_images(
        dict([
            make_img_bboxes_map(image_datum_aro) 
            for image_datum_aro in data['items']
        ])
    )