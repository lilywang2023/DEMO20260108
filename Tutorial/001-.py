def example_function(arg1, *args, kwarg1=None, **kwargs):
    print(f"Single positional argument: {arg1}")
    print(f"Additional positional arguments: {args}")
    print(f"Keyword argument kwarg1: {kwarg1}")
    print(f"Additional keyword arguments: {kwargs}")


example_function(1, 2, 3, kwarg1='a', kwarg2='b', kwarg3='c')
