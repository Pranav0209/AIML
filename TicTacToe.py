def initialize_board():
    return [[' ' for _ in range(3)] for _ in range(3)]

def print_board(board):
    print("-------------")
    for row in board:
        print("|", " | ".join(row), "|")
        print("-------------")

def player_move(board, player):
    while True:
            row, col = map(int, input("Enter row and column (0-2) separated by space: ").split())
            if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == ' ':
                board[row][col] = player
                break
            print("Invalid move. Try again.")

def is_winner(board, player):
    return any(
        all(board[i][j] == player for j in range(3)) or
        all(board[j][i] == player for j in range(3))
        for i in range(3)
    ) or all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3))

def is_board_full(board):
    return all(cell != ' ' for row in board for cell in row)

def minimax(board, is_max):
    if is_winner(board, 'X'):
        return 1
    if is_winner(board, 'O'):
        return -1
    if is_board_full(board):
        return 0

    best_score = float('-inf') if is_max else float('inf')
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X' if is_max else 'O'
                score = minimax(board, not is_max)
                board[i][j] = ' '
                best_score = max(score, best_score) if is_max else min(score, best_score)
    return best_score

def find_best_move(board):
    best_score, best_move = float('-inf'), None
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'X'
                score = minimax(board, False)
                board[i][j] = ' '
                if score > best_score:
                    best_score, best_move = score, (i, j)
    return best_move

def main():
    board = initialize_board()
    print_board(board)

    while True:
        player_move(board, 'O')
        print_board(board)
        if is_winner(board, 'O'):
            print("Player O wins!")
            break
        if is_board_full(board):
            print("It's a draw!")
            break

        ai_move = find_best_move(board)
        if ai_move:
            board[ai_move[0]][ai_move[1]] = 'X'
        print_board(board)
        if is_winner(board, 'X'):
            print("AI Player (X) wins!")
            break
        if is_board_full(board):
            print("It's a draw!")
            break

if __name__ == "__main__":
    main()
