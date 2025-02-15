import numpy as np

def hill_climb(func, x, step=0.01, max_iter=1000):
    for _ in range(max_iter):
        current = func(x)
        # Evaluate the function at the neighbors.
        up = func(x + step)
        down = func(x - step)
        
        # If neither neighbor improves the value, stop.
        if up <= current and down <= current:
            break
        
        # Move in the direction of improvement.
        x += step if up >= down else -step
        
    return x

def func(x):
    return -0.2 * x**4 + 1.5 * x**3 - 3 * x**2 + 2 * x + 10 * np.sin(x)

x0 = 78
x_max = hill_climb(func, x0)
print(f"The value of x that maximizes the function is {x_max}.")
