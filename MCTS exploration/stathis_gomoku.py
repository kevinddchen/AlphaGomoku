#
# AI that learns to play Gomoku using
#        reinforcement learning
#                (MCTS)
#

# packages
from copy import deepcopy
import random
from stathis_mcts import *


# use mcts if True, random if False
use_mcts_or_random = True



# Gomoku board class
class Board:
    # create constructor (init board class instance)
    def __init__(self, size, board=None):
        # define size
        self.size = size
        
        # define players
        self.player_1 = 'x'
        self.player_2 = 'o'
        self.empty_square = '.'
        
        # define board position
        self.position = {}
        
        # init (reset) board
        self.init_board()
        
        # create a copy of a previous board state if available
        if board is not None:
            self.__dict__ = deepcopy(board.__dict__)
    
    # init (reset) board
    def init_board(self):
        # loop over board rows
        for row in range(self.size):
            # loop over board columns
            for col in range(self.size):
                # set every board square to empty square
                self.position[row, col] = self.empty_square
    
    # make move
    def make_move(self, row, col):
        # create new board instance that inherits from the current state
        board = Board(self.size, self)
        
        # make move
        board.position[row, col] = self.player_1
        
        # swap players
        (board.player_1, board.player_2) = (board.player_2, board.player_1)
    
        # return new board state
        return board
    
    # get whether the game is drawn
    def is_draw(self):
        # loop over board squares
        for row, col in self.position:
            # empty square is available
            if self.position[row, col] == self.empty_square:
                # this is not a draw
                return False
        
        # by default we return a draw
        return True
    
    # check whether the game is won and if 
    def is_win(self):
        
        
        # if nobody has won (yet), the winner is None
        winner = None
        
        
        ##################################
        # vertical sequence detection
        ##################################
        
        # loop over board columns
        for col in range(self.size):
            # define winning sequence list
            winning_sequence = []
            previous_square_occupant = None
            
            # loop over board rows
            for row in range(self.size):
                # if found same next element in the row
                if self.position[row, col] == previous_square_occupant and previous_square_occupant!=self.empty_square:
                    # update winning sequence
                    winning_sequence.append((row, col))
                else:
                    # start the counting from the start
                    winning_sequence = [ (row, col) ]
                    winner = self.position[row, col]
                    previous_square_occupant = winner
                    
                # if we have 5 consecutive elements
                if len(winning_sequence) == 5:
                    # return the game is won state
                    return True, winner
        
        ##################################
        # horizontal sequence detection
        ##################################
        
        # loop over board rows
        for row in range(self.size):
            # define winning sequence list
            winning_sequence = []
            previous_square_occupant = None
            
            # loop over board columns
            for col in range(self.size):
                # if found same next element in the row
                if self.position[row, col] == previous_square_occupant and previous_square_occupant!=self.empty_square:
                    # update winning sequence
                    winning_sequence.append((row, col))
                else:
                    # start the counting from the start
                    winning_sequence = [ (row, col) ]
                    winner = self.position[row, col]
                    previous_square_occupant = winner
                    
                # if we have 5 consecutive elements
                if len(winning_sequence) == 5:
                    # return the game is won state
                    return True, winner
    
        ##################################
        # 1st diagonal sequence detection
        ##################################
        
        ## loop over start positions in the first column
        for start_row in range(self.size):
            
            # define winning sequence list
            winning_sequence = []
            previous_square_occupant = None
            
            # init column
            row = start_row
            col = 0
            
            while col<self.size and row<self.size:
 
                # if found same next element in the row
                if self.position[row, col] == previous_square_occupant and previous_square_occupant!=self.empty_square:
                    # update winning sequence
                    winning_sequence.append((row, col))
                else:
                    # start the counting from the start
                    winning_sequence = [ (row, col) ]
                    winner = self.position[row, col] 
                    previous_square_occupant = winner
                    
                # if we have 3 elements in the row
                if len(winning_sequence) == 5:
                    # return the game is won state
                    return True, winner
                col += 1
                row += 1
                
                
        ## loop over start positions in the first row        
        for start_col in range(self.size):
            
            # define winning sequence list
            winning_sequence = []
            previous_square_occupant = None
            
            # init column
            row = 0
            col = start_col
            
            while col<self.size and row<self.size:
 
                # if found same next element in the row
                if self.position[row, col] == previous_square_occupant and previous_square_occupant!=self.empty_square:
                    # update winning sequence
                    winning_sequence.append((row, col))
                else:
                    # start the counting from the start
                    winning_sequence = [ (row, col) ]
                    winner = self.position[row, col] 
                    previous_square_occupant = winner
                    
                # if we have 5 elements in the row
                if len(winning_sequence) == 5:
                    # return the game is won state
                    return True, winner
                col += 1
                row += 1                
                

        ##################################
        # 2nd diagonal sequence detection
        ##################################
        
        ## loop over start positions in the last column
        for start_row in range(self.size):
            
            # define winning sequence list
            winning_sequence = []
            previous_square_occupant = None
            
            # init column
            row = start_row
            col = self.size - 1
            
            while -1<col and row<self.size:
 
                # if found same next element in the row
                if self.position[row, col] == previous_square_occupant and previous_square_occupant!=self.empty_square:
                    # update winning sequence
                    winning_sequence.append((row, col))
                else:
                    # start the counting from the start
                    winning_sequence = [ (row, col) ]
                    winner = self.position[row, col] 
                    previous_square_occupant = winner
                    
                # if we have 5 elements in the row
                if len(winning_sequence) == 5:
                    # return the game is won state
                    return True, winner
                col -= 1
                row += 1
                
                
        ## loop over start positions in the first row        
        for start_col in range(self.size):
            
            # define winning sequence list
            winning_sequence = []
            previous_square_occupant = None
            
            # init column
            row = 0
            col = start_col
            
            while -1<col and row<self.size:
 
                # if found same next element in the row
                if self.position[row, col] == previous_square_occupant and previous_square_occupant!=self.empty_square:
                    # update winning sequence
                    winning_sequence.append((row, col))
                else:
                    # start the counting from the start
                    winning_sequence = [ (row, col) ]
                    winner = self.position[row, col] 
                    previous_square_occupant = winner
                    
                # if we have 5 elements in the row
                if len(winning_sequence) == 5:
                    # return the game is won state
                    return True, winner
                col -= 1
                row += 1  
        
        # by default return non winning state
        return False, winner
    
    
    # generate legal moves to play in the current position
    def generate_states(self):
        # define states list (move list - list of available actions to consider)
        actions = []
        
        # loop over board rows
        for row in range(self.size):
            # loop over board columns
            for col in range(self.size):
                # make sure that current square is empty
                if self.position[row, col] == self.empty_square:
                    # append available action/board state to action list
                    #actions.append( str(row+1)+','+str(col+1) )
                    if use_mcts_or_random:
                        actions.append(self.make_move(row, col))
                    else: 
                        actions.append( [row, col] )
                    



        
        # return the list of available actions (board class instances)
        return actions
    
    # main game loop
    def game_loop(self):
        print('\n  Gomoku by Stathis \n')
        print('  Type "exit" to quit the game')
        print('  Move format [x,y]: 1,2 where 1 is row and 2 is column')
        
        # print board
        print(self)
        
        if use_mcts_or_random:                                           #       mcts = MCTS()
            # create MCTS instance
            mcts = MCTS()                                    
        
                
        # game loop
        while True:
            # get user input
            user_input = input('> ')
        
            # escape condition
            if user_input == 'exit': break
            
            # skip empty input
            if user_input == '': continue
            
#             try:
            # parse user input (move format [col, row]: 1,2) 
            row = int(user_input.split(',')[0]) - 1
            col = int(user_input.split(',')[1]) - 1

            # check move legality
            if self.position[row, col] != self.empty_square:
                print(' Square already occupied!')
                continue



            # make move on board
            self = self.make_move(row, col)




            # print board
            print(self)




            available_actions = self.generate_states()
            

#             print('available_actions: ', available_actions)

            # search for the best move
            
            if use_mcts_or_random:
                best_move = mcts.search(self)
                print('mcts replies')
            else:
                best_move = random.choice(available_actions)                       #mcts.search(self)



            # legal moves available
            if use_mcts_or_random:
                # make AI move here
                try:
                    self = best_move.board
                except:
                    pass
            else:
                self = self.make_move(  best_move[0], best_move[1] )


            # print board
            print(self)

            # check if the game is won
            won, winner = self.is_win()
            if won:
                print('player "%s" has won the game!\n' % winner )
                break

            # check if the game is a draw
            elif self.is_draw():
                print('Game is drawn!\n')
                break

#             except Exception as e:
#                 print('  Error:', e)
#                 print('  Illegal command!')
#                 print('  Move format [x,y]: 1,2 where 1 is column and 2 is row')

                
                
                
                
                
    # print board state
    def __str__(self):
        # define board string representation
        board_string = ''
        
        # loop over board rows
        for row in range(self.size):
            # loop over board columns
            for col in range(self.size):
                board_string += ' %s' % self.position[row, col]
            
            # print new line every row
            board_string += '\n'
        
#         # prepend side to move
#         if self.player_1 == 'x':
#             board_string = '\n--------------\n "x" to move:\n--------------\n\n' + board_string
        
#         elif self.player_1 == 'o':
#             board_string = '\n--------------\n "o" to move:\n--------------\n\n' + board_string
                        
        # return board string
        return board_string

# main driver
if __name__ == '__main__':
    
    # choose size 
    size = 7
    # create board instance
    board = Board(size)
    
    # start game loop
    board.game_loop()
        
        