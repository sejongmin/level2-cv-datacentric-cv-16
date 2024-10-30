import json 
from pathlib import Path

def create_json_without_(ufo_path, output_path, ):
    with Path(ufo_path).open(encoding='utf8') as f:
        ufo_data = json.load(f)
        