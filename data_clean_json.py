import json 
from pathlib import Path

def check_for_key():
    pass

def create_json_without_(ufo_path: str, output_path: str, key: str=""):
    with Path(ufo_path).open(encoding='utf8') as f:
        ufo_data = json.load(f)
        
    print(type(ufo_data))
    print(ufo_data.keys())
    print(ufo_data['images'].keys())
    print(ufo_data['images']['extractor.vi.in_house.appen_001105_page0001.jpg'].keys())
    print(ufo_data['images']['extractor.vi.in_house.appen_001105_page0001.jpg']['words'].keys())
    print(ufo_data['images']['extractor.vi.in_house.appen_001105_page0001.jpg']['words']['0001'].keys())
    print(ufo_data['images']['extractor.vi.in_house.appen_001105_page0001.jpg']['words']['0001']['transcription'])
    
    

if __name__=="__main__":
    create_json_without_(ufo_path="data/vietnamese_receipt/ufo/train.json", output_path="./result.json", key="")

    