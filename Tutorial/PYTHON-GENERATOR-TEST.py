def natural_numbers():
    n = 1
    while True:
        yield n
        n += 1

# 不会耗尽内存！
gen = natural_numbers()
print(next(gen))  # 1
print(next(gen))  # 2


def countdown(n):
    while n > 0:
        yield n
        n -= 1

for x in countdown(3):
    print(x)  # 3, 2, 1


class Countdown:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n + 1


for x in Countdown(9):
    print(x)  # 3, 2, 1


from dis import dis

def f():
    a = 10
    yield a
    b = 2
    yield b
    c = a**b
    yield c

    print("c=",c)
    print("END")

# d= dis(f)

def echo():
    while True:
        received = yield
        print("Got:", received)

g = echo()
next(g)  # 启动生成器（“预激”）
g.send("hello")  # 需要先调用 next() 或 send(None)