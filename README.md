

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
모델 학습에 필요한 하이퍼파라미터는 train.sh와 args.py에서 확인할 수 있습니다. 

<br/>
<br/>

# 기본 추론 코드 실행
```
python inference.py
```
모델 추론에 필요한 하이퍼파라미터는 test.sh와 args.py에서 확인할 수 있습니다. 

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
- 
```

```
### inference.py
- 

### dataset.py
- 

### deteval.py
- 
```

```
### ensemble.py
- 
```

```
### data_clean_json.py
- 
```

```
### create_annotation.py, create_annotation.ipynb
- 
```
```

### google_image_translate_with_selenium.py
- 
```
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
- 
```
```
### ufo_to_datumaro/
-
```

```
### wandb_code/
- 
```

```
### anno/
- 
```

```
