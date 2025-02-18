import numpy as np
# Get function input from the user
func_str = input("Enter a function of x: ")
# Convert the string into a lambda function
f = lambda x: eval(func_str)
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
# Initial guess
x0 = float(input("Enter the initial x value: "))
# Perform hill climbing
x_max = hill_climb(f, x0)
print(f"The value of x that maximizes the function is {x_max}.")
