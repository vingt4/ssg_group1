# 심층신경망개론 Final Project Group1
## 📜프로젝트 소개
해당 프로젝트는 심장질환 조기 진단을 위한 딥러닝 모델을 활용한 심전도(ECG) 데이터 분석 프로젝트입니다. 해당 프로젝트는 다양한 딥러닝 모델을 비교하고 분석하여, 정확도를 높이고 모델을 경량화하는 것에 초점을 맞췄습니다. 주요 분석 모델로는 Vision Transformer (ViT), DeiT, EfficientNetV2, ConVNexT 4가지 딥러닝 모델을 사용하였습니다. 또한 해당 네 가지 모델을 사용하여 ECG데이터 분석을 통한 심장질환 분류 과제를 수행한 후 하이퍼 파라미터 변환 등을 통하여 성능을 높이고, 성능을 유지하면서 모델을 가지치기, 양자화 등을 통해 경량화시켰습니다.
## 🧑🏻‍👩🏻‍👦🏼팀원
구교현, 박효정, 변예원, 윤태준, 최은혁
## Related Research
### 1. HeartBEiT: Vision Transformer for Electrocardiogram Data Improves Diagnostic Performance at Low Sample Sizes
### 연구 내용
HeartBEiT 모델은 Vision Transformer(ViT) 아키텍처를 기반으로 설계된 심전도(ECG) 데이터 분석 모델로, 특히 적은 양의 레이블된 데이터 환경에서 높은 성능을 발휘하는 것이 특징입니다. 해당 모델은 12개의 트랜스포머 블록으로 구성되어 있으며, 각 블록은 768개의 히든 레이어와 12개의 어텐션 헤드를 포함합니다. 또한 ViT 모델의 핵심적인 구조를 계승하면서도, ECG 데이터 분석에 특화된 형태로 발전되었습니다.

해당 연구에서는 Masked Image Modeling(MIM)을 도입하여 사전 학습을 수행했습니다. 이 기법은 자연어 처리의 Masked Language Modeling(MLM)기법을 시각 데이터에 확장한 것으로, 레이블 데이터가 없는 대규모 데이터셋에서도 효과적인 학습을 가능하게 합니다. MIM은 이미지의 일부를 숨긴 채 학습하도록 하여, 모델이 숨겨진 부분을 예측하며 데이터를 학습하는 방식으로 동작합니다.

해당 모델은 레이블된 데이터가 전체 데이터의 1% 수준일 때도 약 0.8 이상의 정확도를 달성하며, 일반 ViT 모델이 동일한 조건에서 기록한 0.6대 정확도를 크게 상회했습니다. 이는 기존 데이터에 레이블 정보를 매핑시키는 과정인 레이블링 비용이 높은 의료 데이터 환경에서 특히 유용하게 적용할 수 있다는 점에서 의의를 가집니다.
### 한계점 
HeartBEiT 모델은 적은 양의 레이블된 데이터로도 높은 성능을 발휘할 수 있다는 장점이 있지만, 대규모 비레이블 데이터셋을 활용한 사전 학습이 필수적이라는 한계가 있습니다. 이는 모델이 높은 복잡도를 가지며, 이에 따른 연산 비용이 매우 크다는 문제를 동반합니다. 또한, 비레이블 데이터의 품질이 학습 결과에 미치는 영향에 대한 추가 연구가 필요합니다.

### 2. Automated ECG Image Classification with InceptionV3
### 연구 내용
해당 연구는 Google에서 제안한 Convolutional Neural Network(CNN) 모델인 InceptionV3를 기반으로, ECG 데이터를 이미지로 변환하여 분류 작업에 활용한 연구입니다. InceptionV3는 여러 크기의 필터를 병렬로 적용하여 다양한 특성을 추출하는 것이 특징인 모델입니다. 이를 ECG 이미지 분류 작업에 적합하도록 fine-tuning하였습니다. 해당 모델은 PTB-XL 데이터셋을 기반으로 ECG 파형 분류에 대한 학습을 진행하는 한편, ECG-Image-Kit을 사용하여 잡음 등의 다양한 artifact들을 포함한 합성 이미지를 제작하여 해당 모델의 fine-tuning 과정에 활용하였습니다. 

해당 모델은 EfficientNetB0 모델과 비교하여 더 낮은 F-score와 높은 GPU time을 기록하였습니다. 하지만 해당 모델은 EfficientNetB0와 비교하여 주최 측의 test dataset에서 더욱 높은 F-score를 획득하였습니다. 이는 EfficientNetB0 모델과 비교하여 해당 연구의 InceptionV3 모델이 더욱 높은 일반화 성능을 보였음을 의미합니다. 또한 해당 결과에서 유의미하게 봐야할 점은 ViT 기반의 Swin-Tiny 모델의 결과입니다. 해당 모델은 가장 높은 GPU time을 기록하여 높은 연산 비용을 나타냈음에도 가장 낮은 F-score결과값을 기록하였습니다. 이는 Vision Transformer 기반의 모델의 경우 모델의 복잡성에서 기인한 더욱 세부적인 fine-tuning 과정이 필요함을 보여줍니다.

### 한계점
InceptionV3 기반의 접근법은 CNN 모델 특유의 연산 효율성 덕분에 실용적인 성능을 보였지만, 최신 Vision Transformer 모델(ViT, BEiT) 기반 접근법과 비교했을 때 정확도가 낮은 편입니다. 또한, 이미지화 과정에서 추가적인 연산 비용이 발생하며, 다량의 레이블된 데이터가 필요한 점도 주요 한계로 지적됩니다.

### 3. Heart Disease Detection using Vision-Based Transformer Models from ECG Images
### 연구 내용
해당 연구는 ECG 데이터를 이미지로 변환하고 Vision Transformer 모델을 활용하여 심장 질환을 탐지하는 방법론을 제안하였습니다. 해당 연구는 특정한 모델을 발전시킨 연구가 아닌 현재 제시된 다양한 ViT 모델들을 ECG 데이터에 맞게 Fine-Tuning하여 다양한 모델의 성과를 정리하였습니다. 

Google ViT의 경우 전체적으로 가장 높은 성능을 기록했으며, 특히 장기적 맥락 정보를 잘 활용하였습니다. Microsoft 사의 Swin Transformer의 경우 Google ViT 보다는 낮은 성능을 기록하였으나 데이터의 지역적 특징 학습에서 강점을 보였습니다. 하지만 전역적인 전체 이미지의 맥락을 처리하는 데에 있어서는 약간의 한계점을 보였습니다. 그러나 이러한 다양한 ViT 기반 모델들은 모두 충분한 컴퓨팅 자원을 필요로 한다는 단점이 있습니다. 이러한 상황에서 CNN 기반의 모델들은 적절한 정확도와 더불어 낮은 계산 복잡도를 보여 실시간 환경이 중요한 병원에서의 사용 가능성을 높였습니다. 정리하였을 때, 다양한 모델을 활용하여 PTB-XL 데이터셋을 사용하여 부정맥, ST-T 변화 등 다양한 심장 질환을 탐지한 결과, AUC 0.96과 F1 Score 0.82를 기록하며 기존의 CNN 기반 모델을 능가하는 성능을 보였습니다. 특히 그 중에서도 Vision Transformer 모델은 데이터가 제한적인 환경에서도 높은 일반화 성능을 보여주었습니다

### 한계점
해당 연구에서 제시된 다양한 접근법들은 모델의 연산 비용이 높아 실제 임상 환경에서 적용하기 어려운 문제가 있습니다. 또한, ECG 데이터를 이미지로 변환하는 과정에서 발생하는 정보 손실 가능성에 대한 추가 연구가 필요합니다.

### 4. Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL
### 연구 내용
해당 연구는 PTB-XL 데이터셋을 기반으로 ECG 데이터를 분석하기 위한 다양한 딥러닝 모델(CNN, RNN)의 성능을 비교하고 벤치마킹을 수행한 연구입니다. ResNet, Inception 모델 등 CNN 기반 아키텍처를 사용하여 ECG 데이터의 분류 작업을 수행하였으며, Data Augmentation(데이터 증강)기법을 활용하여 모델의 일반화 성능을 향상시켰다는 특징이 있었습니다. 또한 해당 연구에서는 모델의 해석 가능성을 높이기 위해서 불확실성 분석, 그리고 Grad-CAM 기법을 연구에 적용하였습니다.

연구 결과, ResNet 기반 모델은 F1 Score 0.81, AUC 0.95 이상의 성능을 기록하며 ECG 데이터 분석에서 매우 높은 정확도를 보여주었습니다. 대부분의 작업에서 가장 높은 성능을 기록했으며 연구 과정에서 ECG 데이터 분석에 적합한 모델로 평가되었습니다. Inception 기반 모델의 경우 F1 Score: 0.79, AUC: 0.93 결과값을 기록하며 ResNet보다 약간 낮은 성능을 보였으나, 데이터 증강 기법 적용 시 성능이 크게 향상되는 모습을 보였습니다.

### 한계점
해당 연구는 CNN 모델만을 활용하여 긴 시계열 정보를 충분하게 학습하지 못할 수 있다는 한계점이 존재합니다. 이에 따라 다양한 ViT 기반 모델들을 활용한 연구 발전의 필요성이 나타납니다.

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


## ViT(Vision Transformer)
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
## DeiT(Data efficient image Transformer)
DeiT 모델 설명

### <모델 개요>
DeiT(Data efficient image Transformer)는 ViT 모델의 변형형입니다. DeiT 모델은 기존 ViT 모델이 대규모 데이터셋이 확복되어야 효과적으로 학습할 수 있는 문제를 개선하여, 소규모 데이터셋으로도 높은 성능을 나타낼 수 있도록 설계되었습니다. 이러한 DeiT 모델은 기존의 ViT 모델의 기본적인 구조를 공유하며, Knowledge Distillation 기법을 사용하여 기존 ViT 모델이 가지고 있었던 단점을 개선하였습니다. 본 프로젝트에서는 DeiT-tiny, DeiT-base 등의 모델을 사용했으며, PTB-XL 데이터셋을 활용한 ECG 데이터를 기반으로 심장질환 분류 태스크를 수행했습니다.

### <구조 및 주요 구성 요소>
DeiT는 ViT 모델과 대부분의 구조를 공유하고 있기 때문에, DeiT 모델에서 특징적으로 존재하는 구성 요소를 기반으로 설명합니다.

1. 지식 증류(Knowledge Distillation)
DeiT는 ViT 모델과 비교하여 지식 증류 기법을 적용하였다는 점에서 차이가 있습니다.
  - 지식 증류란 크고 복잡한 구조를 가진 모델로, 높은 성능을 발휘하지만 계산량이 많고 비효율적일 수 있는 Teacher 모델과 비교적 작고 Teacher 모델의 지식을 학습하여 계산 효율성을 높이면서도 성능을 유지하도록 설계된 Student 모델이 함께 존재합니다.
  - 이 과정에서 데이터의 실제 정답 레이블인 Hard Target과 Teacher 모델이 예측한 레이블에 대한 확률 분포인 Soft target이 등장하며, 지식 증류 기법에서는 이 두가지 target 값을 결합하여 학습에 활용합니다.
  - 즉, Hard Target에 대한 손실 함수에서 이미 실제 레이블에 대한 예측 오류를 최소화하는 데에 초점을 맞추고 있으므로, Student 모델에서는 Teacher Model에서 예측한 확률 분포와의 차이를 줄이는 데에만 집중하면 되는 것입니다.
  - 손실 함수는 Total Loss = α * Hard Loss + (1−α) * Soft Loss의 형태로 구성되며 α는 Hard Loss와 Soft Loss 간의 가중치를 조절하는 하이퍼파라미터입니다.
  - 이러한 지식 증류기법은 규모가 크고 복잡한 Teacher 모델이 아닌 Student 모델의 학습을 중점적으로 진행하여 모델의 경량화를 실현했습니다. 또한 Soft Target의 다양한 확률 분포 데이터를 바탕으로 기존의 데이터보다 더욱 풍부한 데이터를 학습할 수 있어 더욱 적은 데이터 양으로도 충분한 학습이 가능해졌습니다.
    
2. Distillation Token
  - Distillation Token은 Teacher 모델에서 생성되는 soft target(확률 분포)를 학습하는 데에 사용합니다.
  - Distillation Token은 이러한 soft target 값을 더욱 효과적으로 활용할 수 있게 하며 이는 DeiT 모델의 일반화 성능을 향상시킵니다.
  - 이 또한 ViT 모델의 핵심인 Self-Attention 매커니즘을 활용하여 처리됩니다. 

3. Distillation Head
 - Distillation Head 위에서 언급한 Distillation Token을 처리하기 위해 설계된 새로운 구성 요소입니다.
 - 기존 모델이 Classification Head만 존재했던 반면, DeiT는 Distillation Token을 활용하기 위한 Distillation Head를 추가하였습니다.
 - Distillation Head는 Distillation Token에서 주어지는 soft target 값을 기반으로 모델을 학습합니다.
 - 결국 DeiT 모델은 이러한 차이점을 바탕으로 Classification Head와 Distillation Head의 출력값을 모두 사용하여 학습하며, 최종 과정에서는 두 출력값의 평균을 계산하여 최종 예측값을 도출합니다.
     
### <하이퍼파라미터 최적화>

- 패치 크기: (16x16)
- 임베딩 차원: 192
- Transformer 깊이: 12
- 헤드 수: 3
- MLP 히든 레이어 차원: 768 (임베딩 차원 × 4)
- 드롭아웃 비율: 0.2

### <학습 및 평가>
DeiT 모델은 BCEWithLogitsLoss를 손실 함수로 사용하며 Adam 옵티마이저를 활용하여 학습을 진행했습니다. 조기 종료를 통해 과적합을 방지하며, 검증 데이터에서 최적 성능을 기록한 모델을 체크포인트로 저장합니다. 

- 평가지표:
  다중 클래스 분류 문제를 위해 Accuracy, F1-Score(매크로 평균)를 주요 평가지표로 사용하였습니다.
- 검증 및 테스트 결과:
  - 학습 및 검증 데이터에서 안정적인 Accuracy와 F1-Score를 기록하였으며 테스트 데이터에서도 비슷한 결과를 나타냈습니다.
  - 신기하게도 DeiT-tiny와 Deit-base 중 더욱 경량화를 목적으로 만들어진 tiny 모델에서 더욱 좋은 성능이 나타났습니다.
  

## EfficientNetV2
EfficientNetV2 모델 설명

### <모델 개요>
EfficientNetV2는 CNN(Convolutional Neural Network) 기반의 모델로, 기존의 EfficientNet 모델의 성능과 효율성을 개선한 버전입니다. 이 모델은 컴퓨터 비전 task에서 더욱 빠르고 정확한 추론을 목표로 하며, NAS(Neural Architecture Search)와 Progressive Learning 전략으로 이미지 크기에 따른 정규화를 조정해나가는 점진적인 학습을 진행하며 정확도와 효율성을 극대화합니다. 
본 프로젝트에서는 기본적인 아키텍처를 지닌 EfficientNetV2-S 모델을 사용하여 ECG 데이터를 기반으로 심장질환 분류를 수행했습니다.

### <구조 및 주요 구성 요소>
1. 입력 및 초기 레이어 : 첫 번째 Conv2d 레이어는 3×3 필터와 stride=2를 사용하며, 활성화 함수로 SiLU를 사용해 성능을 최적화했습니다.
2. MBConv와 Fused-MBConv 블록
- MBConv: 기존 EfficientNet의 핵심 구성 요소로, MobileNet 구조에서 발전된 Mobile Inverted Bottleneck Convolution을 채택. 깊이별 연산인 Depthwise Convolution과 Squeeze and Excittation 블록으로 채널 간 관계를 모델링하여 중요한 특징을 강조
- Fused-MBConv: 초기 레이어에서 사용되며, 계산 효율성 향상. 또한 Swish라는 부드러운 형태의 활성화 함수를 사용해 학습 안정성과 성능 향상에 기여
3. Progressive Learning : 학습 초기에 작은 이미지 크기로 시작하여 점진적으로 크기를 증가시킴으로써, 복잡한 패턴 학습을 통해 학습 시간 단축 및 성능 유지에 기여
4. Efficient Scaling : 복합 스케일링(Compound Scaling)을 통해 네트워크 깊이(Depth), 너비(Width), 해상도(Resolution)를 조정하여 효율성과 성능을 극대화
5. Stochastic Depth Regularization : 각 블록에 대해 확률적으로 레이어를 드롭하여 모델의 일반화 성능을 강화
6. Adaptive Average Pooling : 전역 풀링을 통해 공간적 크기를 줄이고, 최종 출력 채널에 연결

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
test

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

