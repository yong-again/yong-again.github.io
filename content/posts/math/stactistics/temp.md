---
title: 1. Basic Concepts of Statistics
menu:
  sidebar:
    name: 1. Basic Concepts of Statistics
    identifier: math-stats # <-- 자식의 고유 ID
    parent: math-statistics-root
    weight: 300
---

<aside>
💡 수치형 특성이 두 값의 범위에 안에 놓이도록 스케일$^{(scale)}$을 바꿔야 합니다.

</aside>

**`해결`**  사이킷런의 MinMaxScaler를 사용해 특성 배열의 스케일을 조정합니다.

```python
# Import Library
import numpy as np
from sklearn import preprocessing

# Make Feature
feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])

# Make Scaler Object
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Transfrom Feature sacle
scaled_feature = minmax_scale.fit_transform(feature)

# Print Feature
scaled_Feature

>>>
array([[0.        ],
       [0.28571429],
       [0.35714286],
       [0.42857143],
       [1.        ]])
```

**`설명`** 최소-최대 스케일링$^{(min-max \; scaling)}$은 특성의 최솟값과 최댓값을 활용하여 일정범위 안으로 값을 조정$^{(보통 \; 0에서1, \; -1에서 1 사이)}$

$$
x_i^\prime = \frac{x_i -min(x)}{max(x)-min(x)} \\
$$

$x$는 특성 벡터, $x_i$는 특성 $x$의 개별원소, $x^\prime_i$는 스케일된 원소를 말합니다. 
사이킷런의 `MinMaxScaler`는 특성 스케일을 위해 두 가지 방법을 제공합니다. 첫번째로 `fit` 메서드를 사용해 특성의 최솟값과 최댓값을 계산한 다음 `transform` 메서드로 특성의 스케일을 조정합니다. 

두 번째로, `fit_transform` 메서드로 두 연산을 한번에 처리합니다. 이 둘 사이에 계산상 차이는 없습니다. 동일한 변환을 다른 데이터 세트에 적용하려면 `fit` 메서드와 `transform` 메서드를 따로 호출 해야합니다.

💡덧붙임

훈련 세트와 테스트 세트의 스케일을 따로 조정하면 안됩니다. 예를 들면 훈련 세트의 스케일을 조정하고자 구한 최솟값과 최댓값을 사용하여 테스트 세트를 변환해야 합니다. 그 이유를 간단한 예를 통해서 알아 보겠습니다. 

샘플 중 처음 세 개를 훈련 세트, 나머지 두 개를 테스트 세트라고 가정해 보겠습니다. 먼저 두 세트를 독립적으로 각각 변환합니다.

```python
# Make Feature
feature = np.array([[-500.5],
                    [-100.1],
                    [0],
                    [100.1],
                    [900.9]])

# Transform Train Data set
preprocessing.MinMaxScaler().fit_transform(feature[:3])

>>>
array([[0. ],
       [0.8],
       [1. ]])

# Transform Test Data set
preprocessing.MinMaxScaler().fit_transform(feature[3:])

>>>
array([[0.],
       [1.]])
```

훈련 세트와 테스트 세트를 가각 변환하면 서로 다른 비율로 데이터를 변환합니다. 훈련세트에 있는 0과 테스트 세트에 있는 900.9가 모두 1로 바뀌었습니다. 

이번에는 훈련 세트에서 학습한 변환기로 테스트 세트를 학습해보겠습니다.

```python
# Scale transform training to train data set
scaler = preprocessing.MinMaxScaler().fit(feature[:3])
scaler.transform(feature[:3])

>>>
array([[0. ],
       [0.8],
       [1. ]])

# using trained scaler test data set transform 
scaler.transform(feature[3:])

>>>
array([[1.2],
       [2.8]])
```

훈련 세트를 학습한 변환기 객체를 사용하여 원본 데이터셋과 동일한 비율로 테스트 세트를 변환했습니다.

그 결과 서로 다른 비율로 데이터를 변환하게 된다면 스케일링 후 데이터가 왜곡됩니다.