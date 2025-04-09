# Four Connect Game Simulation

This repository contains two Python implementations of a board game simulation based on the *m-n-k* game (a generalization of Connect Four). Both implementations allow interactive gameplay and performance evaluation on different board configurations.

## Project Structure
```
Project folder/
├─ Plots/
├─ four_connect.py
├─ four_connect_pruned.py
```

## File Overview

- **four_connect.py**  
  Implements the game using a standard minimax algorithm. It includes functions to initialize the board, display it, simulate moves, and recursively evaluate game outcomes. The code also measures the number of states visited and the time taken for the first move.

- **four_connect_pruned.py**  
  Enhances the basic implementation by incorporating alpha-beta pruning into the minimax algorithm. This optimization reduces the number of nodes evaluated in the search tree, resulting in faster move recommendations and improved performance.

- **Plots/**  
    Contains the generated plots from the performance metrics of both implementations. The plots visualize the execution time and number of states visited during the evaluation process.

## Features

- **Interactive Gameplay:**  
  Both implementations offer a command-line interface where you can play against an AI opponent. The board updates after every move, and the AI provides recommended moves based on its evaluation.

- **State Evaluation:**  
  The game detects terminal conditions (win/draw) by analyzing rows, columns, and diagonals for sequences of symbols matching the win condition (`k` in a row).

- **Performance Metrics:**  
  Functions are provided to measure the execution time and the number of states visited during move evaluations. These metrics are visualized using tables created with Matplotlib.

## How to Run

- **Standard Minimax Version:**

Run the following command to start the interactive game using the standard minimax implementation:

python four_connect.py

- **Alpha-Beta Pruned Version:**

Run the following command to launch the game with alpha-beta pruning:

python four_connect_pruned.py

Follow the on-screen prompts to enter your moves. The board will update accordingly, and the AI will recommend the best move based on its evaluation.

## Dependencies

You can install the required dependencies with:

pip install requirements.txt




