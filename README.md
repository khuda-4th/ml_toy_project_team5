# 💻ml_toy_project_team5
주제: **객체 탐지**를 통한 **결제 자동화 시스템**<br>
서비스명: 마트알바<br>

|팀원|역할|
|:-------:|-----------------------------------------------|
|권주명|데이터 전처리, 모델링 , ppt(데이터 전처리)|
|김건형|모델링, ppt(하이퍼파라미터튜닝, 결과), 코드 정리|
|박상영|모델링, ppt(모델), 발표|
|양유경|데이터 전처리, ppt(데이터 eda)|
|이준용|모델링, ppt(데이터 eda)|
|조수현|데이터 전처리, ppt(평가),  ppt 정리(미리캔버스)|
|최호윤|데이터 전처리, ppt(데이터 전처리, 주제 설명), 발표|

## 📜Data
원본 데이터: 서울특별시_상품 표지 이미지 AI 학습 데이터셋<br>
https://www.data.go.kr/data/15081991/fileData.do <br>
<br>

데이터 전처리: yolo style에 맞게 bounding box를 변형해주었다. 방법은 아래와 같다.<br>
<img width="467" alt="image" src="https://github.com/khuda-4th/ml_toy_project_team5/assets/112493995/92aec928-6a8e-4a05-9fb4-0b53dfac090c">

## 🤖YOLO(You Only Look Once)
Object Detection: 이미지나 비디오에서 객체를 식별하고 찾는 컴퓨터 비전 작업<br>
YOLO: 최첨단 실시간 객체 탐지 시스템으로, 빠르고 정확한 데이터 처리 속도를 가진다.<br>

YOLO 모델을 학습하는 방법은 다음 코드와 같이 ultralytics 모듈을 이용하면 매우 간단하다.<br>
```python
from ultralytics import YOLO
import os
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH = '경로'
model = YOLO('yolov8n.pt')
model.train(data=PATH + '/data.yaml', epochs=500,
            patience=20, batch=64, imgsz=1024, device=device)

print(len(model.names), model.names)
```

## 🧑‍💻Result
YOLO 모델을 학습한 후 best.pt로 predict를 해 본 결과, 아래 사진과 같이 장바구니의 상품들을 어느정도 식별함을 알 수 있다. (가격은 임의로 설정하였다.)<br>
<br>
<img src="https://github.com/khuda-4th/ml_toy_project_team5/assets/112493995/273fe268-0d7b-4d34-a0be-1a4000315e9a" width="350" height="400"/>
<br>
<img src="https://github.com/khuda-4th/ml_toy_project_team5/assets/112493995/d28f6aa0-0a5e-418c-afbc-0efee597787f" width="500" height="150"/>
<br>
**아쉬운 점:** <br>
주어진 훈련 데이터가 상품을 정면으로 찍은 모습 밖에 없었기 때문에 여러 각도에서 촬영한 상품을 잘 인식하지 못했다.<br>
YOLO 모델로 높은 정확도의 모델을 구현하려면 완성도 높은 학습 데이터가 필요하다.<br>
<br>
<br>
완성도 높은 학습 데이터는 다음과 같다.<br>
https://aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=realm&dataSetSn=64 <br>
용량이 700GB가 넘어 학습하는 데 오랜 시간이 걸리겠지만, 시도해봤으면 좋겠다.<br>
