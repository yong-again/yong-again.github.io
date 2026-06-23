---
title: 3. Standard Scaler
menu:
  sidebar:
    name: 3. Standard Scaler
    identifier: ml-cookbook-standard-scaler # <-- 자식의 고유 ID
    parent:  ml-cookbook             # <-- 부모의 이름표(identifier)를 여기에 적습니다.
    weight: 10
---

---

<aside>
💡 특성을 평균이 0이고 표준편차가 1이 되도록 변환해야 합니다.

</aside>

`해결`  사이킷런의 StandardScaler를 사용하여 두 변환을 모두 수행 할 수 있습니다.

```python
# Import Library
import numpy as np
from sklearn import preprocessing

# Make Feature
x = np.array([[-1000.1],
              [-200.2],
              [500.5],
              [600.6],
              [9000.9]])
reshape_x = x.reshape(-1,1)

# Make Scaler Object
scaler = preprocessing.StandardScaler()

# Feature Transform
standardized = scaler.fit_transform(reshape_x)
standardized
>>>
array([[-0.76058269],
       [-0.54177196],
       [-0.35009716],
       [-0.32271504],
       [ 1.97516685]])
```

`설명` 

이 방식은 표준화를 사용하여 데이터의 평균 $\bar x$가 0이고 표준편차 $\sigma$가 1이 되도록 변환합니다.

$$
x^\prime_i = \frac{x_i-\bar x}{\sigma}
$$

$x^\prime_i$는 $x_i$의 표준화된 상태입니다. 변환된 특성은 원본 값이 특성 평균에서 몇 표준편차만큼 떨어져 있는지로 표현합니다.(통계학에서는 z-score)

min-max scaling 보다 많이 쓰이지만 학습 알고리즘에 특화 되어있습니다. 예를들어 주성분 분석$^{(principal \; component \; analiysis)}$은 표준화가 적합하지만 신경망$^{(neural \; network)}$에는 min-max scaling을 종종 권장합니다. 일반적으로 다른 방법을 사용할 특별한 이유가 없으면 표준화를 추천합니다.

위 출력결과에서 평균과 표준편차를 구해 표준화의 효과를 살펴보겠습니다.

```python
# print average & std
print('average: ', round(standardized.mean()))
print('standard deviation: ', standardized.std())

>>>
average:  0
standard deviation:  1.0
```

데이터에 이상치가 많다면 중간값과 사분위 범위를 사용하여 특성을 표준화하는 것이 좋습니다. 이상치가 평균과 표준치에 부정적 영향을 끼치기 때문입니다. 사이킷런의 `RobustScaler` 가 이런 방법을 제공합니다.

```python
# Make Transform object
robust_scaler = preprocessing.RobustScaler()

# Feature transform
robust_scaler.fit_transform(x)

>>>
array([[-1.87387612],
       [-0.875     ],
       [ 0.        ],
       [ 0.125     ],
       [10.61488511]])
```

**💡 덧붙임** 

데이터를 오름차순으로 나열했을 때 75%에 위치한 값(3사분위수)과 25%에 위치한 값(1사분위수)의 차를 사분위범위$^{(interquatile \; range)}$(IQR)이라고 부릅니다. `Robustscaler` 는 데이터에서 중간값을 빼고 IQR로 나눕니다. 

```python
interquatile_range = x[3] - x[1]
(x - np.median(x)) / interquatile_range

>>>
array([[-1.87387612],
       [-0.875     ],
       [ 0.        ],
       [ 0.125     ],
       [10.61488511]])
```

`QuantileTransformer` 는 훈련 데이터를 1,000개의 분위로 나누어 0~1 사이에 고르게 분포시킴으로 이상치로 인한 영향을 줄입니다.