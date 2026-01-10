
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



def execute():
    g = f()
    print("got generator",g)

    try:
        aa = next(g)
        print("aa=",aa)

        bb = next(g)
        print("bb=",bb)

        cc = next(g)
        print("cc=",cc)

        dd = next(g)
        print("dd=",dd)

    except StopIteration:
        print("GENERATOR END")

execute()
