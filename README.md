

# 1. Project Overview (프로젝트 개요)
- 다국어 영수증 OCR

![image](https://github.com/user-attachments/assets/e866601e-0938-4b0d-bc66-4f1eda497abf)

- 카메라로 영수증을 인식할 경우 자동으로 영수증 내용이 입력되는 어플리케이션이 있습니다. 이처럼 OCR (Optical Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

- OCR은 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

본 대회에서는 다국어 (중국어, 일본어, 태국어, 베트남어)로 작성된 영수증 이미지에 대한 OCR task를 수행합니다.

본 대회에서는 글자 검출만을 수행합니다. 즉, 이미지에서 어떤 위치에 글자가 있는지를 예측하는 모델을 제작합니다.

본 대회는 제출된 예측 (prediction) 파일로 평가합니다.

모델의 입출력 형식은 다음과 같습니다.

입력 : 글자가 포함된 JPG 이미지 (학습 총 400장, 테스트 총 120장)

출력 : bbox 좌표가 포함된 UFO Format (상세 제출 형식은 Overview > Metric 탭 및 강의 6강 참조)

<br/>
<br/>

# 2. Team Members (팀원 및 팀 소개)
| 곽기훈 | 김재환 | 양호철 | 오종민 | 조소윤 | 홍유향 |
|:------:|:------:|:------:|:------:|:------:|:------:|
| <img src="https://github.com/user-attachments/assets/fb56b1d0-9c5c-49c0-a274-f5b7ff7ab8b1" alt="곽기훈" width="150"> | <img src="https://github.com/user-attachments/assets/28a7109b-4959-473c-a6e4-5ee736370ab6" alt="김재환" width="150"> | <img src="https://github.com/user-attachments/assets/9007ffff-765c-4ffa-80bf-31668fe199ba" alt="양호철" width="150"> | <img src="https://github.com/user-attachments/assets/8760f7bd-10d8-4397-952b-f1ca562b90d4" alt="오종민" width="150"> | <img src="https://github.com/user-attachments/assets/22baca4a-189a-4bc3-ab1c-8f6256637a16" alt="조소윤" width="150"> | <img src="https://github.com/user-attachments/assets/91f96db7-3137-42d2-9175-8a55f1493b31" alt="홍유향" width="150"> |
| T7102 | T7128 | T7204 | T7207 | T7252 | T7267 |
| [GitHub](https://github.com/kkh090) | [GitHub](https://github.com/Ja2Hw) | [GitHub](https://github.com/hocheol0303) | [GitHub](https://github.com/sejongmin) | [GitHub](https://github.com/whthdbs03) | [GitHub](https://github.com/hyanghyanging) | 

<br/>
<br/>

# 개발 환경 및 버젼
```
python==3.10.14
pip install -r requirements.txt
pip install streamlit
pip install easyocr
pip install wandb
pip3 install selenium
pip3 install webdriver-manager
sudo apt-get install xvfb
pip install PyVirtualDisplay
```

<br/>
<br/>

# 기본 학습 코드 실행
```
python train.py
```

<br/>
<br/>

# 기본 추론 코드 실행
```
python inference.py
```
모델 추론 결과로 csv파일을 반환합니다. 

<br/>
<br/>

# Project Structure (프로젝트 구조)
```plaintext
📦level2-cv-datacentric-cv-16
 ┣ 📂.git
 ┣ 📂.github
 ┃ ┣ 📂ISSUE_TEMPLATE
 ┃ ┃ ┗ 📜-title----body.md
 ┃ ┣ 📜.keep
 ┃ ┗ 📜pull_request_template.md
 ┣ 📂anno
 ┃ ┣ 📂1
 ┃ ┣ 📂2
 ┃ ┣ 📂3
 ┃ ┣ 📂cord
 ┃ ┗ 📂crawl
 ┣ 📂augmentation
 ┃ ┗ 📜augmentation.ipynb
 ┣ 📂crawling
 ┃ ┗ 📜realwhth.ipynb
 ┣ 📂streamlit
 ┃ ┣ 📜뭐뭐.py
 ┃ ┗ 📜뭐뭐.ipynb
 ┣ 📂ufo_to_datumaro
 ┃ ┣ 📜뭐뭐.py
 ┃ ┗ 📜뭐뭐.ipynb
 ┣ 📂wandb_code
 ┃ ┣ 📜뭐뭐.py
 ┃ ┗ 📜뭐뭐.ipynb
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┗ 📜뭐뭐.py
```

<br/>

### train.py
- 학습을 하기 위한 기본적인 코드입니다. OCR 중 객체 위치 탐지만을 학습합니다.
```
python train.py
```
### inference.py
- 학습된 객체 위치 탐지 모델으로 객체 위치를 추정합니다. 결과값으로 CSV파일이 저장됩니다. 
```
python inference.py
```
### dataset.py
- dataset을 정의하며, 데이터에 적용될augmentation을 다룹니다.
```

```
### deteval.py
- 
```

```
### ensemble.py
- 추론을 통해 얻은 csv파일을 hard voting 방식으로 앙상블합니다. 
```
python ensemble.py --input_dir your_csv_dir
# --output_dir 결과 출력 위치
# --iou_min 실험할 iou 최소값
# --iou_max 실험할 iou 최대값
# --iou_step 실험할 때 iou 증가량
# --vote_min 실험할 때 bbox를 결정할 때 vote 최소 개수
# --vote_max 실험할 때 bbox를 결정할 때 vote 최대 개수
# --single_iou 단일 iou 값
# --single_vote 단일 vote 값
```
### data_clean_json.py
- UFO format annotation 파일에서 transcription 기준으로 bbox를 삭제합니다.
```
python data_clean_json.py
```
### create_annotation.py, create_annotation.ipynb
- easyocr를 활용하여 입력으로 들어오는 이미지에 대해 annotation을 수행한 후 ufo format으로 변환합니다.
```
python create_annotation.py
```

### google_image_translate_with_selenium.py
- google translate 웹사이트에서 지원하는 image tranlsation을 활용하여 자동으로 이미지를 특정 언어로 번역하고 자동 annotation까지 수행한 후 바로 활용 가능한 데이터셋으로 만들어줍니다.
```
python google_image_translate_with_selenium.py
```

### make_cord.ipynb
- 
```

```

### crawling/realwhth.ipynb
- 
```

```
### streamlit/
- streamlit을 활용하여 데이터 및 모델 출력물에 대해 시각화할 수 있습니다.
```
load_json.py
main.py
visualize_csv.py
visualize_json.py
```
### ufo_to_datumaro/
-
```
bbox_to_polygon.py
datumaro_to_ufo_v2.ipynb
ufo_done.py
ufo_to_datumaro_v1.ipynb
ufo_to_datumaro_v2.ipynb
utils.py
utils_v2.py
```
### wandb_code/
- 
```
inference_wandb.py
logger_epoch.py
logger_sweep.py
sweep.yaml
train_wandb.py
train_wandb_onecycleLR.py
train_wandb_sweep.py
train_wandb_v2.py
```
### anno/
- 

