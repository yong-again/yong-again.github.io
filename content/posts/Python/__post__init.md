---
title : __post_init__ 메서드 이해하기
menu:
    sidebar:
        name: __post_init__
        identifier: post-init
        parent: python
        weight: 20
---


- dataclass에서 사용되는 메소드
- 객체의 모든 필드가 초기화 된 직후 “후처리” 작업을 위해 사용됨
- 이름 그대로 초기화(`__init__` ) 이후에 실행되는 로직을 담기 위한 기능

## `__post_init__`의 의미

- 데이터 클래스는 `__init__` 매서드를 자동으로 생성해줌. → 필드를 초기화할 필요가 없음
- 하지만, 필드 초기화 후 작업이 필요한 경우가 발생
    - 초기화된 값이 유효한지 검증하고 싶을 때
    - 초기화된 값을 조합하여 새로운 속성을 만들고 싶을 때
- 실행 순서
    - 객체 생성 시 전달된 인자들로 `@dataclass` 가 자동 생성한 `__init__`이 먼저 실행됨
    - 모든 필드 초기화 완료
    - `__post_init__` 메서드가 자동으로 호출됨

---

## `__post_init__` 의 주요 기능과 예시

1. 초기화 값 검증 
    
    객체가 생성될 때 전달된 값들이 특정 조건을 만족하는지 검사할 수 있음
    

```python
from dataclasses import dataclass

@dataclass
class User:
    username: str
    age: int

    def __post_init__(self):
        print(f"✅ __post_init__ 호출: {self.username}의 나이를 검증합니다.")
        if self.age <= 0:
            raise ValueError("나이는 0보다 커야 합니다.")

# 정상적인 경우
user1 = User(username="Alice", age=30)
# >> ✅ __post_init__ 호출: Alice의 나이를 검증합니다.

# 비정상적인 경우
try:
    user2 = User(username="Bob", age=-5)
except ValueError as e:
    print(e)
# >> ✅ __post_init__ 호출: Bob의 나이를 검증합니다.
# >> 나이는 0보다 커야 합니다.
```