import json 

def create_json_without_(ufo_path, output_path, ):
    with open(ufo_path, 'r') as f:
        ufo_data = json.load(f)
        