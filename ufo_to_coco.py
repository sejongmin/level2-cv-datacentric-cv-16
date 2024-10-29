import json

def ufo_to_coco(ufo_path, coco_output_path, category_name="text"):
    with open(ufo_path, 'r') as f:
        ufo_data = json.load(f)
    
    # Initialize COCO format structure
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Define a category for OCR tasks
    category_id = 1
    coco_data["categories"].append({
        "id": category_id,
        "name": category_name,
        "supercategory": "OCR"
    })
    
    # Initialize counters
    image_id = 0
    annotation_id = 0
    
    # Process each image in UFO data
    for image_name, image_data in ufo_data['images'].items():
        # Add image info to COCO format
        img_entry = {
            "id": image_id,
            "file_name": image_name,
            "width": image_data.get("img_w", 0),
            "height": image_data.get("img_h", 0)
        }
        coco_data["images"].append(img_entry)
        
        # Process each word in the image
        for word_id, word_data in image_data["words"].items():
            points = word_data["points"]
            
            # Calculate bounding box as [xmin, ymin, width, height]
            xmin = min(p[0] for p in points)
            ymin = min(p[1] for p in points)
            xmax = max(p[0] for p in points)
            ymax = max(p[1] for p in points)
            width = xmax - xmin
            height = ymax - ymin

            # Convert points to a single list for segmentation
            segmentation = [coord for point in points for coord in point]
            
            # Add annotation info
            annotation_entry = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "iscrowd": 0,
                "segmentation": [segmentation]
            }
            coco_data["annotations"].append(annotation_entry)
            annotation_id += 1
        
        image_id += 1

    # Write to COCO output JSON file
    with open(coco_output_path, 'w') as f:
        json.dump(coco_data, f)