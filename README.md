

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
lanms==1.0.2
opencv-python==4.10.0.84
shapely==2.0.5
albumentations==1.4.12
torch==2.1.0
tqdm==4.66.5
albucore==0.0.13
annotated-types==0.7.0
contourpy==1.1.1
cycler==0.12.1
eval_type_backport==0.2.0
filelock==3.15.4
fonttools==4.53.1
fsspec==2024.6.1
imageio==2.35.0
importlib_resources==6.4.2
Jinja2==3.1.4
kiwisolver==1.4.5
lazy_loader==0.4
MarkupSafe==2.1.5
matplotlib==3.7.5
mpmath==1.3.0
networkx==3.1
numpy==1.24.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.18.1
nvidia-nvjitlink-cu12==12.6.20
nvidia-nvtx-cu12==12.1.105
packaging==24.1
pillow==10.4.0
pydantic==2.8.2
pydantic_core==2.20.1
pyparsing==3.1.2
python-dateutil==2.9.0.post0
PyWavelets==1.4.1
PyYAML==6.0.2
scikit-image==0.21.0
scipy==1.10.1
six==1.16.0
sympy==1.13.2
tifffile==2023.7.10
tomli==2.0.1
triton==2.1.0
typing_extensions==4.12.2
numba==0.60.0
```

<br/>
<br/>

# 학습 코드 실행
```
sh train.sh
```
모델 학습에 필요한 하이퍼파라미터는 train.sh와 args.py에서 확인할 수 있습니다. 

<br/>
<br/>

# 추론 코드 실행
```
sh test.sh
```
모델 추론에 필요한 하이퍼파라미터는 test.sh와 args.py에서 확인할 수 있습니다. 

<br/>
<br/>

# Project Structure (프로젝트 구조)
```plaintext
📦level1-imageclassification-cv-16
 ┣ 📂.github
 ┃ ┣ 📂ISSUE_TEMPLATE
 ┃ ┃ ┗ 📜-title----body.md
 ┃ ┣ 📜.keep
 ┃ ┗ 📜pull_request_template.md
 ┣ 📂model
 ┃ ┣ 📜cnn.py
 ┃ ┣ 📜mlp.py
 ┃ ┣ 📜model_selection.py
 ┃ ┣ 📜resnet18.py
 ┃ ┣ 📜timm.py
 ┃ ┗ 📜torchvision_model.py
 ┣ 📂util
 ┃ ┣ 📜augmentation.py
 ┃ ┣ 📜checkpoints.py
 ┃ ┣ 📜data.py
 ┃ ┣ 📜losses.py
 ┃ ┣ 📜metrics.py
 ┃ ┣ 📜optimizers.py
 ┃ ┗ 📜schedulers.py
 ┣ 📜.gitignore
 ┣ 📜README.md
 ┣ 📜args.py
 ┣ 📜eda.ipynb
 ┣ 📜eda.py
 ┣ 📜erase_dot_files.py
 ┣ 📜gradcam.py
 ┣ 📜image_augmentation.py
 ┣ 📜separate.py
 ┣ 📜test.py
 ┣ 📜test.sh
 ┣ 📜train.ipynb
 ┣ 📜train.py
 ┣ 📜train.sh
 ┗ 📜trainer.py
```

<br/>

### train.sh
- train.py 파일을 실행시키면서 학습에 필요한 인자를 입력하는 쉘 스크립트 파일. 학습 재개 시 저장 시점과 동일한 하이퍼파라미터를 사용
```
 --mode: train 모드, test 모드 있음. train.sh에선 train 고정
 --device: cpu, gpu 선택
 --data_root: data 디렉터리 고정
 --csv_path: train(+validation) 데이터셋 파일 경로 설정
 --val_csv: 사용x
 --height, --width: 학습 데이터셋의 Resize 크기 결정
 --num_classes: class 개수 입력
 --auto_split: 사용x
 --split_seed: train_test_split의 random state seed 값 설정
 --stratify: train_test_split의 비율을 고정하는 기준이 될 column 결정
 --model: 사용할 모델명 기입. timm의 경우 timm-model_name 형태로 입력하면 timm 라이브러리의 모델을 불러옴
 --lr: 학습률 설정
 --lr_scheduler: 스케줄러 선택
 --lr_scheduler_gamma: stepLR, RduceLROnPlateau의 learning rate decay 감소 비율을 지정하는 파라미터
 --lr_scheduler_epochs_per_decay: stepLR의 lr 감소 주기 설정
 --batch: 배치 사이즈
 --loss: loss function 선택
 --optim: 옵티마이저 선택
 --r_epochs: train set과 validation set의 크기를 바꾸기 시작하는 에포크 설정 (뒤에서 n번째부터 시작)
 --seed: random값의 기준 설정
 --transform: 사용할 augmentation 클래스(라이브러리 기준으로 나눔)를 선택
 --augmentations: 사용할 augmentation 기법을 설정. "_"로 split하여 string 분리
 --adjust_ratio: 이미지의 종횡비를 1:1로 맞춤
 --eraly_stopping: 개선이 있는지 감시할 epoch 수 설정. 이 epoch동안 validation accuracy의 개선이 없으면 학습 중단
 --verbose: tqdm 사용 여부 결정. 주석 풀면 True, 아니면 False
 --resume, --checkpoint_path: 체크포인트에 저장된 모델 불러오기 여부, 체크포인트.pt 파일 경로. 세트로 사용
```
### train.py
- trainer.py의 trainer 클래스를 불러와서 학습 시킴

### test.sh, test.py
- test.sh에서 인자를 받아 test.py 파일을 실행해 test data의 예측 결과 저장. train.sh와 비슷

### trainer.py
- 학습 모듈
```
 -create_config_txt : train.sh 호출 당시 내용을 checkpoint 폴더에 함께 저장하여 어떤 하이퍼파라미터를 사용했는지 기록
 -save_checkpoint_tmp : 이전 fold(or epoch)와 비교하여 validation accuracy가 1% 이상 개선되면 checkpoint 저장
 -final_save_model : 이전 accuracy와 관계 없이 마지막 모델 저장
 -train_epoch : 모델학습 1 epoch 수행
 -validate : 모델 검증 수행
 -train : epoch만큼 학습하는 함수. train.sh를 통해 전달받은 resome 파라미터가 true이면 self.load_settings 함수로 checkpoint 모델을 불러옴
 -k_fold_train : train 함수에 K-Fold Cross Validation을 적용함
 -load_settings 체크포인트 저장 시점의 모델과 optimizer, scheduler 등 학습에 필요한 정보를 불러옴
```
### eda.py
- 모든 데이터의 메타데이터를 추출하여 csv파일로 만드는 파일

### args.py
- train.sh, test.sh에서 받아온 인자를 파이썬에서 사용할 수 있는 변수로 변환하는 모듈

### gradcam.py 
- Grad-CAM을 통해 Heatmap을 반환하는 함수를 포함하는 파일

### image_augmentation.py
- offline augmentation하는 파일. 종횡비를 맞추기 위해 흰 배경 추가하는 코드와 flip을 적용하는 코드가 있다. 추가된 이미지를 포함한 ./data/train1.csv 파일을 생성

### separate.py
- 데이터셋을 물리적으로 분리하는 파일

### util/augmentation.py
- augmentation 라이브러리를 관리하는 모듈. Albumentation을 사용
```
 -AlbumentationsTransforms 클래스: train.sh에서 받는 augmentations 인자를 가지고 클래스의 생성자가 full_aug_list를 보고 aug_list에 추가하여 사용할 증강 기법을 선택
 -TransformSelector: train.sh에서 받은 transform 인자로 어떤 증강 클래스를 사용할지 선택
```
### util/checkpoints.py
- 체크포인트를 저장/불러오기 하는 모듈

### util/data.py
- Dataset, DataLoader를 재정의하는 모듈
```
 -CustomDataset 클래스: 대회를 위해 제공받은 데이터셋에 맞게 데이터를 불러오게하는 Dataset
 -HoDataset, HoDataLoader 클래스: K-Fold cross validation을 위한 Dataset, DataLoader
```
### util/losses.py
- loss function을 가짐

### util/metrics.py
- f1 score을 계산하는 모듈

### util/optimizers.py
- train.sh의 optim 인자를 받아서 optimizer를 선택할 수 있게 매핑하는 모듈

### util/schedulers.py
- train.sh의 lr_scheduler 인자를 받아서 learning rate scheduler를 선택할 수 있게 매핑하는 모듈

### model/
- model_selection 파일은 다른 모델을 불러오는 파일. timm, torchvision_model은 라이브러리를 쉽게 불러오기 위한 모듈
