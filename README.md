# 심층신경망개론 Final Project Group1
## 📜프로젝트 소개
해당 프로젝트는 심장질환 조기 진단을 위한 딥러닝 모델을 활용한 심전도(ECG) 데이터 분석 프로젝트입니다. 해당 프로젝트는 다양한 딥러닝 모델을 비교하고 분석하여, 정확도를 높이고 모델을 경량화하는 것에 초점을 맞췄습니다. 주요 분석 모델로는 Vision Transformer (ViT), DeiT, EfficientNetV2, ConVNexT 4가지 딥러닝 모델을 사용하였습니다. 또한 해당 네 가지 모델을 사용하여 ECG데이터 분석을 통한 심장질환 분류 과제를 수행한 후 하이퍼 파라미터 변환 등을 통하여 성능을 높이고, 성능을 유지하면서 모델을 가지치기, 양자화 등을 통해 경량화시켰습니다.
## 🧑🏻‍👩🏻‍👦🏼팀원
구교현, 박효정, 변예원, 윤태준, 최은혁
## Data Download
다음의 링크에서 Data.zip 파일을 다운받은 후 같은 폴더에 압축을 해제하면 됩니다.
[Data](https://drive.google.com/file/d/1XbVp8MEL8JADnAY4fpOoWrWMCmsYewJ7/view?usp=sharing)](https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip)
## Data Preprocessing
### 1. 데이터 설명
ptbxl_database.csv: ECG 데이터에 대한 정보를 담은 CSV 파일입니다.

scp_statements.csv: 진단 정보를 포함한 데이터베이스 CSV 파일입니다.

records100 : 100Hz 속도로 샘플링된 ECG 파일입니다.

records500 : 500Hz 속도로 샘플링된 ECG 파일입니다.

### 2. 진단 클래스 매핑
각 ECG 샘플의 scp_codes를 진단 클래스(diagnostic_class)로 매핑합니다. 진단 클래스는 scp_statements.csv의 diagnostic_class 열에서 정의되며, 진단과 관련된 (diagnostic == 1) 항목만 사용하였습니다. 

### 3. 데이터 필터링
매핑된 진단 클래스가 없는 샘플은 제거하였습니다. 또한 각 샘플은 diagnostic_superclass에 해당하는 진단 클래스의 리스트로 저정하였습니다.

### 4. ECG 신호 로드
샘플링 속도(100Hz 또는 500Hz)에 따라 ECG 신호 데이터를 로드합니다. WFDB 모듈을 사용해 ECG 신호와 메타 데이터를 불러오며 신호데이터를 Numpy 배열로 변환하였습니다.

### 5. 데이터셋 분리
데이터셋은 strat_fold 열을 기준으로 나뉩니다:
Train Set: strat_fold가 테스트(10) 및 검증(9) 폴드가 아닌 샘플.
Validation Set: strat_fold가 검증(9) 폴드인 샘플.
Test Set: strat_fold가 테스트(10) 폴드인 샘플.

### 6. 다중 라벨 이진화
진단 클래스는 5개의 Multi Label로 되어있기 때문에 MultiLabelBinarizer를 사용해 라벨 데이터를 이진화하였습니다. 또한 fit_transform으로 학습 및 변환, 이후 검증 및 테스트 데이터를 동일하게 변환하였습니다.

### 7. 데이터셋 클래스 정의
PyTorch의 Dataset을 사용하여 ECGDataset 클래스를 정의하였습니다. 또한 Label은 torch.tensor로 변환하였습니다.

### 8. 데이터 증강 및 변환
torchvision.transforms를 활용하여 데이터를 텐서로 변환하고 크기를 조정(182x256)하였습니다. CNN 모델 사용을 위해 1채널 데이터를 3채널로 확장하였으며, 데이터를 정규화하였습니다.

### 9. DataLoader
DataLoader를 통해 배치 크기 32로 셔플된 데이터를 사용하였습니다.


## Vit
## Deit
## EfficientNetV2
## ConVNext
## 경량화
### 가지치기
```
import torch.nn.utils.prune as prune

# 가중치 40퍼센트 pruning
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.4)
# 가중치 고정 
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        prune.remove(module, 'weight')
```
### 양자화
```
# dynamic quantization 적용
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
# static quantizaion 적용
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibration 및 Quantization 적용
for images, _ in dataloader:
    model(images)
torch.quantization.convert(model, inplace=True)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

#Quantizaion-aware training 적용 하여 모델 훈련
model.train()
for epoch in range(num_epochs):
    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Quantization 적용
model.eval()
torch.quantization.convert(model, inplace=True)
```

