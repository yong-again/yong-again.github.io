---
title: Decorator

menu:
    sidebar:
        name: Decorator
        parent: python  # <-- 이 ID가 부모의 이름표입니다.
        weight: 200
---

- 데코레이터 : Wrapping과 비슷! → 기존의 작성된 함수 및 클래스 를 수정하지 않고, 그 위에 추가적인 기능을 덧씌우거나 감싸서 새로운 기능을 부여하는것.

---

## 데코레이터의 역할

- 어떤 함수(클래스)를 받아서, 기능을 추가한 새로운 함수(클래스)를 돌려주는 함수

1. 원래함수

```python
def say_hello():
	print("Hello!)
```

1. 데코레이터 만들기 

```python
def add_emojis_decorator(original_function):
    # 'original_function' (여기서는 say_hello)을 인자로 받음

    def wrapper():
        print("✨🎉✨")  # 1. 원래 함수 실행 전에 기능 추가
        original_function() # 2. 원래 함수 실행
        print("✨🎉✨")  # 3. 원래 함수 실행 후에 기능 추가
    
    return wrapper # 4. 기능이 추가된 새로운 함수(wrapper)를 반환
```

1. 데코레이터 적용

```python
@add_emojis_decorator
def say_hello():
    print("안녕하세요!")

# 이제 say_hello()를 호출하면...
say_hello()
```

실행결과:

```python
✨🎉✨
안녕하세요!
✨🎉✨
```

`@` 기호는 아래 코드를 단축한 표현이라고 보면 된다!

```python
say_hello = add_emojis_decorator(say_hello)
```