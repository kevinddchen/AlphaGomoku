import numpy as np

class Gomoku:
    '''Gomoku board.

    Parameters ==========

        size <int> ... Dimension of board. Defaults to size=15.
        win <int> ... Number of stones in a chain to win. Defaults to win=5.

    Variables ===========

        size <int>
        win <int>
        board <array> ... Numerical representation of the board; 0=empty, 1=black, -1=white.
          
            Uses matrix convention, not Go convention.
               0 1 2 3 x
            0 |_|_|_|_| ...
            1 |_|_|_|_| ...
            2 |_|_|_|_| ...
            3 |_|_|_|_| ...
            y  . . . . 

        episode <list of tuples (x, y)> ... Moves played so far.
        finished <boolean> ... 'True' if game is finished, 'False' otherwise.
        winner <int> ... '1' if black won, '-1' if white won, '0' otherwise.
        curr_player <int> ... Current player.

    Methods =============

        play(x, y) ... Play a move at coordinates (x, y). Automatically alternates between black and white.
            
        available_actions() ... Returns 2D array of boolean values indicating valid moves. 
            
        available_actions_list() ... Returns 1D array of valid moves, in flattened form: move = x*size + y. 
            
        forbidden_actions()
        
        forbidden_actions_list()

        find_winner(x, y) ... Computes and returns the 'winner' variable, based on most recent move (x, y).

        show() ... Print board. 
            
        copy() ... Returns copied instance of Gomoku game.
    '''

    def __init__(self, size=15, win=5):
        self.size = size
        self.win = win
        self.board = np.zeros((size, size), dtype=np.int8)
        self.episode = []
        self.finished = False
        self.winner = 0
        self.curr_player = +1
        
    def play(self, x, y):
        assert not self.finished, "game has ended"
        assert self.board[x, y] == 0, "invalid move"
        self.board[x, y] = self.curr_player
        self.episode.append((x, y))
        self.curr_player *= -1
        ## game ends when all spaces filled
        if np.sum(self.available_actions()) == 0:
            self.finished = True
        ## game ends when there is a winner
        self.winner = self.find_winner(x, y)
        if self.winner != 0:
            self.finished = True

    def available_actions(self):
        return self.board == 0
    
    def available_actions_list(self):
        return np.where(self.available_actions().flatten())[0]
    
    def forbidden_actions(self):
        return self.board != 0
    
    def forbidden_actions_list(self):
        return np.where(self.forbidden_actions().flatten())[0]

    def find_winner(self, x, y):
        ## look in all directions to see if (x, y) is contained in a 5-chain.
        piece = self.board[x, y]
        ## left-right
        i, j = 0, 0
        while x-i-1 >= 0 and self.board[x-i-1, y] == piece: i += 1
        while x+j+1 < self.size and self.board[x+j+1, y] == piece: j += 1
        if i+j+1 >= self.win: return piece
        ## up-down
        i, j = 0, 0
        while y-i-1 >= 0 and self.board[x, y-i-1] == piece: i += 1
        while y+j+1 < self.size and self.board[x, y+j+1] == piece: j += 1
        if i+j+1 >= self.win: return piece
        ## NW-SE
        i, j = 0, 0
        while x-i-1 >= 0 and y-i-1 >= 0 and self.board[x-i-1, y-i-1] == piece: i += 1
        while x+j+1 < self.size and y+j+1 < self.size and self.board[x+j+1, y+j+1] == piece: j += 1
        if i+j+1 >= self.win: return piece
        ## NE-SW
        i, j = 0, 0
        while x+i+1 < self.size and y-i-1 >= 0 and self.board[x+i+1, y-i-1] == piece: i += 1
        while x-j-1 >= 0 and y+j+1 < self.size and self.board[x-j-1, y+j+1] == piece: j += 1
        if i+j+1 >= self.win: return piece
        return 0

    def show(self):
        pieces = {0:'.', 1:'\u25CF', -1:'\u25CB'}
        colors = {0: "none", 1:"black", -1:"white"}
        ## print recent moves
        if len(self.episode) >= 2:
            print("{0:s} played {1}.".format(colors[self.curr_player], self.episode[-2]))
        if len(self.episode) >= 1:
            print("{0:s} played {1}.".format(colors[-self.curr_player], self.episode[-1]))
        ## print if game has ended
        if self.finished:
            print("game has ended. winner: {0:s}".format(colors[self.winner]))
        else:
            print("{0:s}'s turn.".format(colors[self.curr_player]))
        ## print board
        print("  ", end='')
        for x in range(self.size):
            print("{0:2d}".format(x), end='')
        print()
        for y in range(self.size):
            print("{0:2d}".format(y), end=' ')
            for x in range(self.size):
                print(pieces[self.board[x, y]], end=' ')
            print()
            
    def copy(self):
        new_game = Gomoku(size=self.size, win=self.win)
        new_game.board = self.board.copy()
        new_game.episode = self.episode[:]
        new_game.finished = self.finished
        new_game.winner = self.winner
        new_game.curr_player = self.curr_player
        return new_game



class Player:
    '''Basic player interface.

    Parameters/Variables =========

        name <string> ... Name of player.
        piece <int> ... Either +1 for black or -1 for white.

    Methods ============

        play(game) ... Given Gomoku game, returns move (x, y) to play.
    '''
    def __init__(self, name, piece):
        self.name = name
        self.piece = piece

    def play(self, game):
        ## make random move.
        avail_acts = game.available_actions_list()
        move = np.random.choice(avail_acts)
        return move//game.size, move%game.size

