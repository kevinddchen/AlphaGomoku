import numpy as np

class Gomoku:
    '''Gomoku board.

    Parameters ==========

        size: int. Dimension of board. Default size=15.

    Variables ===========

        size: int.
        board: array. Representation of the board. 0=empty, 1=first player, -1=second player.
          
            Uses matrix convention, not Go convention.
               0 1 2 3 x
            0 |_|_|_|_| ...
            1 |_|_|_|_| ...
            2 |_|_|_|_| ...
            3 |_|_|_|_| ...
            y  . . . . 

        finished: boolean. 'True' if game is finished, 'False' otherwise.
        winner: int. '1' if first player won, '-1' if second player won, '0' otherwise.

    Methods =============

        available_actions() -> 15x15 array
            Returns array of boolean values indicating valid moves. 

        play(x: int, y: int)
            Play a move/action at coordinates (x, y). Automatically alternates
            between first and second players.

        find_winner(x: int, y: int) -> int
            Computes and returns the 'winner' variable from most recent move (x, y).

        print()
            Print board. 
    '''

    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)
        self.finished = False
        self.winner = 0
        self._curr_player = +1

    def available_actions(self):
        return self.board == 0

    def play(self, x, y):
        assert not self.finished, "game has ended"
        assert self.board[x, y] == 0, "invalid move"
        self.board[x, y] = self._curr_player
        self._curr_player *= -1
        ## game ends when there is a winner
        self.winner = self.find_winner(x, y)
        if self.winner != 0:
            self.finished = True

    def find_winner(self, x, y):
        piece = self.board[x, y]
        ## look left-right
        i, j = 0, 0
        while x-i >= 1 and self.board[x-i-1, y] == piece:
            i += 1
        while x+j <= self.size-2 and self.board[x+j+1, y] == piece:
            j += 1
        if i+j+1 >= 5:
            return piece
        ## look up-down
        i, j = 0, 0
        while y-i >= 1 and self.board[x, y-i-1] == piece:
            i += 1
        while y+j <= self.size-2 and self.board[x, y+j+1] == piece:
            j += 1
        if i+j+1 >= 5:
            return piece
        ## look NW-SE
        i, j = 0, 0
        while x-i >= 1 and y-i >= 1 and self.board[x-i-1, y-i-1] == piece:
            i += 1
        while x+j <= self.size-2 and y+j <= self.size-2 and self.board[x+j+1, y+j+1] == piece:
            j += 1
        if i+j+1 >= 5:
            return piece
        ## look NE-SW
        i, j = 0, 0
        while x-i >= 1 and y+i <= self.size-2 and self.board[x-i-1, y+i+1] == piece:
            i += 1
        while x+j <= self.size-2 and y-j >= 1 and self.board[x+j+1, y-j-1] == piece:
            j += 1
        if i+j+1 >= 5:
            return piece
        return 0

    def print(self):
        print(self.board.T)



class Player:
    '''Basic player interface.

    Parameters =========

        name: string. Name of player.
        piece: int. Either +1 for first player or -1 for second player.

    Methods ============

        play(x: int, y: int, game: Gomoku) -> (int, int)
            Returns move/action, given previous move (x, y) by opponent and game.

        learn(episode: list)
            Learn from episode.
    '''
    def __init__(self, name, piece):
        self.name = name
        self.piece = piece

    def play(self, x, y, game):
        pass

    def learn(self, episode):
        pass



class RandomPlayer(Player):
    '''Takes random moves; for playtesting.'''
    def __init__(self, *args):
        super().__init__(*args)

    def play(self, x, y, game):
        actions = np.where(game.available_actions().flatten())[0]
        move = np.random.choice(actions)
        return move//game.size, move%game.size



