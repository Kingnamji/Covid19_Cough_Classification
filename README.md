# Covid19_Cough_Classification
데이콘, 음향 데이터 COVID-19 검출 AI 경진대회를 진행하며 작성한 코드들입니다.
(대회 링크 : https://dacon.io/competitions/official/235910/overview/description)

## Summary
- 해당 코드들은 Colab을 기준으로 작성했습니다. 

- 모델은 MFCC feature의 평균 값과 주어진 나머지 feature들을 활용하는 Pycaret, Tabnet과 MFCC feature를 활용하는 2D CNN을 고려했습니다. (최종적으로, Pycaret을 채택했습니다.) 

- Upsampling, Downsampling을 적용해보기도 하고, Data Augmentation을 적용해 실험을 했으나, 오히려 성능이 저하됐습니다.

## Score (F1)
- Public : 0.6129

- Private : 0.60663
