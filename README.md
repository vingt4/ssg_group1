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
EfficientNetV2 모델 설명

### <모델 개요>
EfficientNetV2는 CNN(Convolutional Neural Network) 기반의 모델로, 기존의 EfficientNet 모델의 성능과 효율성을 개선한 버전입니다. 이 모델은 컴퓨터 비전 task에서 더욱 빠르고 정확한 추론을 목표로 하며, NAS(Neural Architecture Search)와 Progressive Learning 전략으로 이미지 크기에 따른 정규화를 조정해나가는 점진적인 학습을 진행하며 정확도와 효율성을 극대화합니다. 본 프로젝트에서는 기본적인 아키텍처를 지닌 EfficientNetV2-S 모델을 사용하여 ECG 데이터를 기반으로 심장질환 분류를 수행했습니다.

### <구조 및 주요 구성 요소>
1. 입력 및 초기 레이어 : 첫 번째 Conv2d 레이어는 3×3 필터와 stride=2를 사용하며, 활성화 함수로 SiLU를 사용해 성능을 최적화했습니다.
2. MBConv와 Fused-MBConv 블록
- MBConv: 기존 EfficientNet의 핵심 구성 요소로, MobileNet 구조에서 발전된 Mobile Inverted Bottleneck Convolution을 채택. 깊이별 연산인 Depthwise Convolution과 Squeeze and Excittation 블록으로 채널 간 관계를 모델링하여 중요한 특징을 강조
- Fused-MBConv: 초기 레이어에서 사용되며, 계산 효율성 향상. 또한 Swish라는 부드러운 형태의 활성화 함수를 사용해 학습 안정성과 성능 향상에 기여
3. Progressive Learning
- 학습 초기에 작은 이미지 크기로 시작하여 점진적으로 크기를 증가시킴으로써, 복잡한 패턴 학습을 통해 학습 시간 단축 및 성능 유지에 기여
4. Efficient Scaling : 복합 스케일링(Compound Scaling)을 통해 네트워크 깊이(Depth), 너비(Width), 해상도(Resolution)를 조정하여 효율성과 성능을 극대화
5. Stochastic Depth Regularization: 각 블록에 대해 확률적으로 레이어를 드롭하여 모델의 일반화 성능을 강화
6. Adaptive Average Pooling: 전역 풀링을 통해 공간적 크기를 줄이고, 최종 출력 채널에 연결

### <가중치 활용>
1. Pretrained Weights : EfficientNet_V2_S_Weights 클래스를 사용해 ImageNet 데이터셋으로 사전 학습된 가중치를 활용
2. 가중치 초기화 : 모든 Conv2d 레이어는 Kaiming Normal 초기화 방식으로, Linear 레이어는 균등분포 초기화 방식으로 설정됨

### <하이퍼파라미터 최적화>
- 모델 버전: EfficientNetV2-S
- 입력 이미지 크기: 224 ×224
- Batch Size: 32
- Optimizer: Adam optimizer
- Learning Rate: 0.001
- 출력 클래스 수 : 5
- 손실 함수(criterion) : BCEWithLogitsLoss
- stochastic_depth_prob(확률적으로 레이어 드롭) : 0.2
- Dropout 비율: 0.2

### <학습 및 평가>
- 학습 전략: 조기 종료(Early Stopping) 적용 여부에 따라 epoch 수를 달리 학습하여 모델 성능을 비교하였고, 검증 데이터 성능을 기준으로 과적합을 방지하며 최적의 모델을 선택했습니다.
  1) Early Stopping 포함 (20 epoch 학습) : F1 스코어의 조기 향상을 확인하여 8 epoch에서 조기 종료하여 과적합 방지
  2) Early Stopping 없이 (10 epoch 학습) : 고정된 epoch 학습을 통해 전체 epoch 동안 성능 변화 추이를 모니터링하였고 Early Stopping 적용의 필요성을 검증하기 위한 비교 실험으로 사용
- 평가 지표 : Accuracy와 F1-Score (Macro Average)를 주요 평가 지표로 설정했습니다.

EfficientNetV2-S 모델은 Early Stopping과 하이퍼파라미터 최적화를 통해 검증 및 테스트 데이터에서 높은 성능을 기록하며, ECG 데이터를 기반으로 한 심장질환 분류 작업에서 정확도와 효율성을 입증했습니다.

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

