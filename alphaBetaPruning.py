def minimax(depth, index, maximizing, values, alpha, beta, max_depth):
    # If we've reached the maximum depth, return the value at the current node.
    if depth == max_depth:
        return values[index]

    if maximizing:
        best = -float('inf')
        for i in range(2):
            val = minimax(depth + 1, index * 2 + i, False, values, alpha, beta, max_depth)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break  # Alpha-Beta Pruning
        return best
    else:
        best = float('inf')
        for i in range(2):
            val = minimax(depth + 1, index * 2 + i, True, values, alpha, beta, max_depth)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break  # Alpha-Beta Pruning
        return best

if __name__ == '__main__':
    # Read input values and maximum depth.
    values = list(map(int, input("Enter values separated by spaces: ").split()))
    max_depth = int(input("Enter the depth: "))
    
    # Start the minimax algorithm from the root node (index 0) as the maximizing player.
    result = minimax(0, 0, True, values, -float('inf'), float('inf'), max_depth)
    print("The optimal value is:", result)
