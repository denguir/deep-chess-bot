# Chess bot
This is a bot that learns chess in two steps:

## Supervised learning
The first step focuses on learning from a Stockfish engine how to evaluate a chess board.
To do this, we downloaded a chess database ([KingBase 2018](https://archive.org/details/KingBase2018) & [KingBase 2019](https://archive.org/details/KingBase2019)) containing games of GM players that we evaluated with Stockfish. Then, we designed a neural network similar to Giraffe (see docs: giraffe) which has been trained to score similarly as the Stockfish engine


## Reinforcement learning (ongoing)
The second step of this project (still ongoing) is to manage to learn further how to play chess by applying
TD-leaf(lambda) algorithm (see docs: giraffe). The idea of this algorithm is to minimize the error between 
Giraffe's board evaluation at a time t and its own evaluation of the board n steps further in the future.
By doing this, we hope that the network keeps temporal consistency on the evaluation of the board. 

## How to use:
__Warning__: All the scripts use all your CPU cores by default, to change this simply set the N_PROC variable of each script
to the number of cores you want to actually use
### Build Stockfish database
A Stockfish database is already available in data/csv. If you want to extend it you can run data_preparation script.
Just make sure to set max_game parameter (the number of games you want to learn from) to the value you desire
### Learn from Stockfish
Use stockfish_training.py script. This will learn from the Stockfish database you have built and store the model
in the model directory
### Self-learning (ongoing)
Use td_leaf_learning.py script. This is currently in progress
### Play actual game
You can visualize a game between two neural networks in the minimax.py script