def f2():
    try:
        yield 1
        try:
            yield 2
            1/0
            yield 3
        except ZeroDivisionError:
            yield 4
            yield 5
            raise
        except:
            yield 6
        yield 7
    except:
        yield 8
    yield 9
    try:
        a = 100
    finally:
        yield 10
    yield 11



print(list(f2()))

def divide(a, b):
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b

divide(10, 0)  # 抛出: ValueError: 除数不能为零

