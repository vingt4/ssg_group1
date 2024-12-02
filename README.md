# 심층신경망개론 Final Project Group1
## 📜프로젝트 소개
해당 프로젝트는 심장질환 조기 진단을 위한 딥러닝 모델을 활용한 심전도(ECG) 데이터 분석 프로젝트입니다. 해당 프로젝트는 다양한 딥러닝 모델을 비교하고 분석하여, 정확도를 높이고 모델을 경량화하는 것에 초점을 맞췄습니다. 주요 분석 모델로는 Vision Transformer (ViT), DeiT, EfficientNetV2, ConVNexT 4가지 딥러닝 모델을 사용하였습니다. 또한 해당 네 가지 모델을 사용하여 ECG데이터 분석을 통한 심장질환 분류 과제를 수행한 후 하이퍼 파라미터 변환 등을 통하여 성능을 높이고, 성능을 유지하면서 모델을 가지치기, 양자화 등을 통해 경량화시켰습니다.
## 🧑🏻‍👩🏻‍👦🏼팀원
구교현, 박효정, 변예원, 윤태준, 최은혁
## Data Download
다음의 링크에서 파일을 다운받은 후 같은 폴더에 압축을 해제하면 됩니다.
[Data](https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip)
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
Vision Transformer (ViT) 모델 설명

### <모델 개요>

Vision Transformer(ViT)는 Transformer 구조를 컴퓨터 비전 문제에 적용한 모델로, 기존 CNN 모델과 달리 패치(patch) 단위로 이미지를 처리하여 전역적이고 효과적인 이미지 특징 추출이 가능합니다. ViT는 입력 이미지를 작은 패치들로 분할한 후, 각 패치를 임베딩 벡터로 변환하고 Transformer 블록을 통해 처리합니다. 본 프로젝트에서는 ViT 모델을 사용하여 ECG 데이터를 기반으로 심장질환을 분류하는 작업을 수행했습니다.

### <구조 및 주요 구성 요소>

1. 패치 임베딩 (Patch Embedding):
   이미지를 고정된 크기의 패치로 분할하고, 각 패치를 저차원 벡터로 임베딩하기 위해 `nn.Conv2d`를 사용하였습니다. 이를 통해 이미지 데이터를 Transformer가 처리할 수 있는 형태로 변환합니다.  

2. 포지셔널 임베딩 (Positional Embedding):
   패치 간의 순서 정보를 Transformer가 학습할 수 있도록, 각 패치에 고유한 포지셔널 벡터를 더해줍니다.  

3. Transformer 블록:
   모델은 여러 개의 ViT 블록으로 구성되며, 각 블록은 다음과 같은 구조를 갖습니다:
   - Layer Normalization: 입력 특징의 분포를 정규화하여 학습을 안정화.
   - Multi-Head Attention: 다양한 시각적 관계를 학습하기 위한 Self-Attention 메커니즘.
   - MLP (Multilayer Perceptron): 비선형 변환을 통해 특징을 학습. 두 개의 Fully Connected Layer와 Dropout으로 구성.
   - 잔차 연결 (Residual Connection): 학습 안정성을 높이고 기울기 소실 문제를 방지.

4. 분류 헤드 (Classification Head):
   Transformer 블록의 출력을 평균 풀링하여 최종적으로 선형 레이어를 사용해 클래스별 출력을 생성합니다.

### <하이퍼파라미터 최적화>

- 패치 크기: (8x8)  
- 임베딩 차원: 256  
- Transformer 깊이: 6  
- 헤드 수: 4 (임베딩 차원 256 / 헤드 차원 64)  
- MLP 히든 레이어 차원: 1024 (임베딩 차원 × 4)  
- 드롭아웃 비율: 0.1  

### <학습 및 평가>

ViT 모델은 BCEWithLogitsLoss를 손실 함수로 사용하며, Adam 옵티마이저와 학습률 0.0001로 학습되었습니다. 조기 종료를 통해 과적합을 방지하며, 검증 데이터에서 최적 성능을 기록한 모델을 체크포인트로 저장합니다. 

- 평가지표:
  다중 클래스 분류 문제를 위해 Accuracy, F1-Score(매크로 평균)를 주요 평가지표로 사용하였습니다.  
- 검증 및 테스트 결과: 
  모델은 검증 및 테스트 데이터에서 높은 성능을 기록하였으며, ECG 데이터 분석에 ViT 구조의 효과를 입증했습니다.

ViT는 심전도(ECG) 데이터와 같은 시계열 이미지 데이터를 처리하는 데 있어 유용한 모델임을 보였으며, 본 프로젝트에서는 ECG 신호를 시각화하여 이미지를 생성한 뒤 ViT를 사용하여 정확도와 효율성을 극대화하였습니다.
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

