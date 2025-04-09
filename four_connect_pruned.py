import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

class Game():
    def __init__(self, m, n, k):
        self.rows = n
        self.columns = m
        self.k = k
        self.states_visited = 0
        self.board = None
        self.initialize_game()

    def initialize_game(self):
        """Initialises the board."""
        self.board = np.full((self.rows, self.columns), "-")
    
    def drawboard(self):
        """Draws the board."""
        for row in self.board:
            print("|", end="")
            for symbol in row:
                if symbol == "-":
                    print("   ", end="")
                else:
                    print(f" {symbol}", end=" ")
            print("|")
    
    def is_valid(self, action):
        """Checks if given column (action) is a valid field
        
        Args:
            action(int): Defines the column that a player put his symbol into
        """
        # Check action is possible given number of columns, and that top position in column isn't filled
        return (0 <= action <= (self.columns-1)) and (self.board[0, action]!="X" and self.board[0, action]!="O")
        
                
    def reset_states_visited(self):
        "Resets visited states count to 0"
        self.states_visited = 0

    
    def get_states_visited(self):
        "Returns how many states have been visited"
        return self.states_visited
    
    
    def check_string(self, sequence):
        "Checks if a sequence contains k number of Xs or Os"
        string_representation = "".join(sequence)
        if "X" * self.k in string_representation:
            return 1
        if "O" * self.k in string_representation:
            return -1
        return None

    def is_terminal(self): 
        "Checks if the match is terminated and if it has returns the respective points"

        # Checking all rows
        for row in range(self.rows):
            result = self.check_string(self.board[row,:])
            if result is not None:
                return result
        
        # Checking all columns 
        for column in range(self.columns):
            result = self.check_string(self.board[:,column])
            if result is not None:
                return result

        # Checking all diagonals
        for offset in range(1-self.rows, self.columns):
            result = self.check_string(np.diagonal(self.board, offset))
            if result is not None:
                return result
            
            result = self.check_string(np.diagonal(np.fliplr(self.board), offset))
            if result is not None:
                return result
        
        # Checking if top row is full with Xs or Os
        top_row = "".join(self.board[0, :])
        if not "-" in top_row:
            return 0

        return None
    
    def simulate_move(self, action, max_turn=True):
        """Simulates how the board looks if a move is made
        
        Args:
            action(int): Defines the column that a player put his symbol into
            max_turn(boolean): Who's turn it is to define what symbol is given
        """
        if max_turn:
            symbol = "X"
        else:
            symbol = "O"
        
        for i in range(self.rows - 1, -1, -1):
            if self.board[i, action] == "-":
                self.board[i, action] = symbol
                return i  # Return the row index where the move was made

        return None

    def max(self, alpha=float('-inf'), beta=float('inf'), depth=9):
        """Implements the max algorithm with a standard depth of 9
        
        Args:
            alpha(int): Defines the alpha used for pruning
            beta(int): Defines beta used for pruning
            depth(int): Until what depth the algorithm should check the tree
        """

        # If depth of 0 is reached, return the state or 0
        if depth == 0:
            return self.is_terminal() or 0, None

        # If current board state is terminal, return value 1 if MAX wins or -1 if MIN wins or 0 for DRAW
        terminal_state = self.is_terminal()
        if terminal_state is not None:
            return terminal_state, None

        # Initialise value list for each action
        v_all = []

        # Iterate through available actions
        for action in range(self.columns):
            v = float("-inf")
            # If action is valid, simulate the resulting state and get MIN to pass min value
            if self.is_valid(action):
                row = self.simulate_move(action, max_turn=True)
                self.states_visited += 1
                v, _ = self.min(alpha, beta, depth=depth-1)
                self.board[row,action] = "-"
            v_all.append(v)

            # if v (alpha) is greater than beta, PRUNE other actions to come
            if v >= beta:
                return (v, action)

            # Update alpha
            alpha = max(alpha, v)

        # Choose action with max value
        recommended_action = np.argmax(v_all)
        chosen_value = max(v_all)
        return (chosen_value, recommended_action)
    

    def min(self, alpha=float('-inf'), beta=float('inf'), depth=9):
        """Implements the min algorithm with a standard depth of 9

        Args:
            alpha(int): Defines the alpha used for pruning
            beta(int): Defines beta used for pruning
            depth(int): Until what depth the algorithm should check the tree
        """

        # If depth of 0 is reached, return the state or 0
        if depth == 0:
            return self.is_terminal() or 0, None

        # If current board state is terminal, return value 1 if MAX wins or -1 if MIN wins or 0 for DRAW
        terminal_state = self.is_terminal()
        if terminal_state is not None:
            return terminal_state, None

        # Initialise value list for each action
        v_all = []

        # Iterate through available actions
        for action in range(self.columns):
            v = float("inf")
            # If action is valid, simulate the resulting state and get MAX to pass max value
            if self.is_valid(action):
                row = self.simulate_move(action, max_turn=False)
                self.states_visited += 1
                v, _ = self.max(alpha, beta, depth=depth-1)
                self.board[row,action] = "-"
            v_all.append(v)

            # if v (alpha) is greater than beta, PRUNE other actions to come
            if v <= alpha:
                return (v, action)

            # update beta
            beta = min(beta, v)

        # Choose action with min value
        recommended_action = np.argmin(v_all)
        chosen_value = min(v_all)
        return (chosen_value, recommended_action)
    
    def play(self):
        "Allows you to play a match of four connect against the min algorithm with recommendation feature"
        self.drawboard()
        action = 1

        # Loop through each players turn while game is still going
        while action is not None:

            # Calculate best action for MAX
            value, action = self.max()
            print(f"Recommended action for MAX is: {action}")
            # Ask for input from user
            action = int(input("What action do you select? "))
            # Update board state
            self.simulate_move(action, max_turn=True)
            self.drawboard()
            self.reset_states_visited()

            # Check if new board state is terminal, print result if so
            result = self.is_terminal()
            if result is not None:
                self.print_result(result)
                break

            # Calculate best action for Min
            value, action = self.min()
            print(f"Recommended action for MIN is: {action}")
            # Update board state
            self.simulate_move(action, max_turn=False)
            self.drawboard()
            self.reset_states_visited()

            # Check if new board state is terminal, print result if so
            result = self.is_terminal()
            if result is not None:
                self.print_result(result)
                break

        pass

    def print_result(self, result):
        """Print the results"""
        if result == 1:
            print('Max wins!')
        elif result == -1:
            print('Min wins!')
        else:
            print('Draw!')

    def time_first_move(self):
        "Times how long it takes to do one move"
        start_time = time.time()
        value, action = self.max()
        end_time = time.time()
        execution_time = end_time - start_time
        return round(execution_time, 3)


def time_experiment():
    """Performs the experiment to see how much time passes / states are visited for different game set-ups"""
    m_times = []
    n_times = []
    k_times = []

    for x in range(3,8):
        game = Game(x,3,3)
        time = game.time_first_move()
        states = game.get_states_visited()
        game.reset_states_visited()
        m_times.append({'m': x, 'n': 3, 'k': 3, 'Time (s)': time, 'States visited': states})
        

        print(m_times)

        game = Game(3,x,3)
        time = game.time_first_move()
        states = game.get_states_visited()
        game.reset_states_visited()
        n_times.append({'m': 3, 'n': x, 'k': 3, 'Time (s)': time, 'States visited': states})

        print(n_times)

    for x in range(1,7):

        game = Game(5, 5, x)
        time = game.time_first_move()
        states = game.get_states_visited()
        game.reset_states_visited()
        k_times.append({'m': 5, 'n':5, 'k': x, 'Time (s)': time, 'States visited': states})

        print(k_times)

    # Create DataFrames
    df_m = pd.DataFrame(m_times)
    df_n = pd.DataFrame(n_times)
    df_k = pd.DataFrame(k_times)

    # Create table figures
    render_table(df_m, 'PRUNED First Move Execution Time for Varying m (columns)')
    render_table(df_n, 'PRUNED First Move Execution Time for Varying n (rows)')
    render_table(df_k, 'PRUNED First Move Execution Time for Varying k (target)')


def render_table(df, title):
    """Creates visual tables for inclusion in our report"""
    # Calculate the number of rows and columns
    row_count, col_count = df.shape

    # Set figure dimensions based on the table size
    fig_width = max(8, col_count * 1.5)  # Width per column
    fig_height = max(2, row_count * 0.6)  # Height per row

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Hide axes
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(14)

    # Adjust table scale to fill the figure
    table.scale(1, 2)  # Adjust these numbers to control scaling

    # Add title to the figure
    plt.title(title, fontsize=16, pad=20)

    # Tight layout to minimize whitespace
    plt.tight_layout()

    plt.show()

    pass




if __name__ == "__main__":
    # time_experiment()
    game = Game(3,3,3)
    game.play()



