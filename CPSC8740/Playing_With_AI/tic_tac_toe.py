def print_board(board):
    print("\n")
    print(" " + board[0] + " | " + board[1] + " | " + board[2] + " ")
    print("---+---+---")
    print(" " + board[3] + " | " + board[4] + " | " + board[5] + " ")
    print("---+---+---")
    print(" " + board[6] + " | " + board[7] + " | " + board[8] + " ")
    print("\n")

def check_win(board):
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8), 
                      (0, 3, 6), (1, 4, 7), (2, 5, 8), 
                      (0, 4, 8), (2, 4, 6)]
    for condition in win_conditions:
        if board[condition[0]] == board[condition[1]] == board[condition[2]] != " ":
            return board[condition[0]]
    if " " not in board:
        return "Tie"
    return False

def play_game():
    board = [" "] * 9
    current_player = "X"

    while True:
        print_board(board)
        
        # Get valid move
        move = None
        while move is None:
            try:
                user_input = int(input(f"Player {current_player}, enter your move (1-9): ")) - 1
                if user_input not in range(9) or board[user_input] != " ":
                    print("Invalid move. Try again.")
                else:
                    move = user_input
            except ValueError:
                print("Please enter a number between 1 and 9.")

        board[move] = current_player

        # Check if game is over
        result = check_win(board)
        if result:
            print_board(board)
            if result == "Tie":
                print("It's a tie!")
            else:
                print(f"Player {result} wins!")
            break

        # Switch player
        current_player = "O" if current_player == "X" else "X"

play_game()
