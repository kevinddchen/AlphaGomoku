import numpy as np

class Gomoku:
    '''Gomoku board.

    Parameters ==========

        size: int. Dimension of board. Default size=15.

    Variables ===========

        size: int.
        board: array. Representation of the board. 0=empty, 1=black, -1=white.
          
            Uses matrix convention, not Go convention.
               0 1 2 3 x
            0 |_|_|_|_| ...
            1 |_|_|_|_| ...
            2 |_|_|_|_| ...
            3 |_|_|_|_| ...
            y  . . . . 

        episode: list of (x, y). Moves played so far.
        finished: boolean. 'True' if game is finished, 'False' otherwise.
        winner: int. '1' if black won, '-1' if white won, '0' otherwise.

    Methods =============

        available_actions() -> array
            Returns array of boolean values indicating valid moves. 

        play(x: int, y: int)
            Play a move at coordinates (x, y). Automatically alternates between black and white.

        find_winner(x: int, y: int) -> int
            Computes and returns the 'winner' variable, based on most recent move (x, y).

        show()
            Print board. 
    '''

    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.episode = []
        self.finished = False
        self.winner = 0
        self._curr_player = +1
        self._moves_left = size*size

    def available_actions(self):
        return self.board == 0

    def play(self, x, y):
        assert not self.finished, "game has ended"
        assert self.board[x, y] == 0, "invalid move"
        self.board[x, y] = self._curr_player
        self.episode.append((x, y))
        self._curr_player *= -1
        self._moves_left -= 1
        ## game ends when all spaces filled
        if self._moves_left == 0:
            self.finished = True
        ## game ends when there is a winner
        self.winner = self.find_winner(x, y)
        if self.winner != 0:
            self.finished = True

    def find_winner(self, x, y):
        ## look in all directions to see if (x, y) is contained in a 5-chain.
        piece = self.board[x, y]
        ## left-right
        i, j = 0, 0
        while x-i-1 >= 0 and self.board[x-i-1, y] == piece: i += 1
        while x+j+1 < self.size and self.board[x+j+1, y] == piece: j += 1
        if i+j+1 >= 5: return piece
        ## up-down
        i, j = 0, 0
        while y-i-1 >= 0 and self.board[x, y-i-1] == piece: i += 1
        while y+j+1 < self.size and self.board[x, y+j+1] == piece: j += 1
        if i+j+1 >= 5: return piece
        ## NW-SE
        i, j = 0, 0
        while x-i-1 >= 0 and y-i-1 >= 0 and self.board[x-i-1, y-i-1] == piece: i += 1
        while x+j+1 < self.size and y+j+1 < self.size and self.board[x+j+1, y+j+1] == piece: j += 1
        if i+j+1 >= 5: return piece
        ## NE-SW
        i, j = 0, 0
        while x+i+1 < self.size and y-i-1 >= 0 and self.board[x+i+1, y-i-1] == piece: i += 1
        while x-j-1 >= 0 and y+j+1 < self.size and self.board[x-j-1, y+j+1] == piece: j += 1
        if i+j+1 >= 5: return piece
        return 0

    def show(self):
        pieces = {0:'.', 1:'\u25CF', -1:'\u25CB'}
        print('  ', end='')
        for x in range(self.size):
            print('{0:2d}'.format(x), end='')
        print()
        for y in range(self.size):
            print('{0:2d}'.format(y), end=' ')
            for x in range(self.size):
                print(pieces[self.board[x, y]], end=' ')
            print()



class Player:
    '''Basic player interface.

    Parameters =========

        name: string. Name of player.
        piece: int. Either +1 for black or -1 for white.

    Methods ============

        play(game: Gomoku) -> (x, y)
            Returns move to play.
    '''
    def __init__(self, name, piece):
        self.name = name
        self.piece = piece

    def play(self, game):
        pass

