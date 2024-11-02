# 스크립트를 실행하면 자동으로 변환을 수행
#python ufo_done_v1.py

import json
from pathlib import Path

def read_json(filename):
    """JSON 파일을 읽어오는 함수"""
    with Path(filename).open(encoding='utf8') as f:
        return json.load(f)

def convert_format(new_test_json, train_json):
    """
    NewTest.json을 train.json 형식으로 변환하는 함수
    파일 이름이 같은 경우 해당 train 데이터의 메타데이터를 사용
    
    Parameters:
    - new_test_json: NewTest.json 데이터
    - train_json: train.json 데이터 (메타데이터 참조용)
    
    Returns:
    - 변환된 JSON 데이터
    """
    result = {"images": {}}
    
    # train.json의 이미지 이름에서 .jpg 제거한 버전 준비
    train_data_map = {
        name.replace('.jpg', ''): data 
        for name, data in train_json["images"].items()
    }
    
    # 기본 메타데이터 (매칭되는 데이터가 없을 경우 사용)
    default_metadata = {
        "chars": {},
        "img_w": 1024,
        "img_h": 1024,
        "num_patches": None,
        "tags": [],
        "relations": {},
        "annotation_log": {
            "worker": "worker",
            "timestamp": "2024-05-30",
            "tool_version": "",
            "source": None
        },
        "license_tag": {
            "usability": True,
            "public": False,
            "commercial": True,
            "type": None,
            "holder": "Upstage"
        }
    }
    
    # NewTest.json의 각 이미지에 대해 처리
    for image_name, image_data in new_test_json["images"].items():
        # train.json에서 매칭되는 메타데이터 찾기
        train_data = train_data_map.get(image_name)
        metadata = {
            "chars": train_data["chars"],
            "img_w": train_data["img_w"],
            "img_h": train_data["img_h"],
            "num_patches": train_data["num_patches"],
            "tags": train_data["tags"],
            "relations": train_data["relations"],
            "annotation_log": train_data["annotation_log"],
            "license_tag": train_data["license_tag"]
        } if train_data else default_metadata
        
        # 새로운 이미지 데이터 구조 생성
        new_image_data = {
            "paragraphs": {},
            "words": {},
            **metadata
        }
        
        # words 정보 복사 (transcription과 points만 유지)
        for word_id, word_data in image_data["words"].items():
            new_image_data["words"][word_id] = {
                "transcription": word_data["transcription"],
                "points": word_data["points"]
            }
        
        # .jpg 확장자를 붙여서 결과에 추가
        result["images"][f"{image_name}.jpg"] = new_image_data
    
    return result

def save_json(data, filename):
    """변환된 JSON을 파일로 저장하는 함수"""
    with Path(filename).open('w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    # JSON 파일 읽기
    new = read_json("/data/ephemeral/home/kjh/level2-cv-datacentric-cv-16/ufo_to_datumaro/datumaro2ufo/ufoTest.json")
    train = read_json("/data/ephemeral/home/kjh/level2-cv-datacentric-cv-16/data/thai_receipt/ufo/train.json")
    
    # 포맷 변환
    converted = convert_format(new, train)
    
    # 결과 저장
    save_json(converted, "converted.json")

if __name__ == "__main__":
    main()