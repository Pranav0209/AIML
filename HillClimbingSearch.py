import numpy as np

def hill_climbing(func, x0, delta=0.01, tol=1e-6, max_iter=1000):
    x = x0
    for _ in range(max_iter):
        x_up, x_down = x + delta, x - delta

        if func(x_up) > func(x):
            x = x_up
        elif func(x_down) > func(x):
            x = x_down
        else:
            break

        if abs(func(x) - func(x0)) < tol:
            break
        x0 = x

    return x

def func(x):
    return -0.2 * x**4 + 1.5 * x**3 - 3 * x**2 + 2 * x + 10 * np.sin(x)

x0 = 78
x_max = hill_climbing(func, x0)
print(f"The value of x that maximizes the function is {x_max}.")
