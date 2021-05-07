import numpy as np
import gomoku

class RandomPlayer(gomoku.Player):
    '''Takes random moves; for playtesting.'''
    def __init__(self, name, piece):
        super().__init__(name, piece)

    def play(self, game):
        actions = np.where(game.available_actions().flatten())[0]
        move = np.random.choice(actions)
        return move//game.size, move%game.size
    

    
class FeaturePlayer(gomoku.Player):
    '''Keeps track of the following features:
     0: Number of 2-chains the move will create.
     1: "       " 3-chains "                  ".
     2: "       " 4-chains "                  ".
     3: "       " 5-chains "                  ".
     4: Number of opponent 2-chains the move will block.
     5: "                " 3-chains "                 ".
     6: "                " 4-chains "                 ".
     7: "                " 5-chains "                 ".
     Points are assigned to each feature according to the w vector.
     Player takes move with maximum points.
    '''
    def __init__(self, name, piece):
        super().__init__(name, piece)
        ## points assigned to each each feature
        self.w = np.array([2, 4, 8, 16, 1, 3, 7, 15])
        
    def play(self, game):
        features = self.get_features(game)
        scores = features.dot(self.w)
        actions = np.where((scores == np.max(scores)).flatten())[0]
        move = np.random.choice(actions)
        return move//game.size, move%game.size
       
    def get_features(self, game):
        ## calculate features as described above -- there are lots of cases to check.
        avail_moves = game.available_actions()
        features = np.zeros((game.size, game.size, 8))
        for x in range(game.size):
            for y in range(game.size):
                if not avail_moves[x, y]:
                    features[x, y] = -1
                    continue
                ## left-right
                i, j = 0, 0
                p_i = game.board[x-1, y] if x-1 >= 0 else 0
                p_j = game.board[x+1, y] if x+1 < game.size else 0
                while p_i != 0 and x-i-1 >= 0 and game.board[x-i-1, y] == p_i: i += 1
                while p_j != 0 and x+j+1 < game.size and game.board[x+j+1, y] == p_j: j += 1
                if p_i == self.piece and p_j == self.piece:
                    features[x, y, min(i+j+1, 5)-2] += 1
                elif p_i == self.piece:
                    features[x, y, min(i+1, 5)-2] += 1
                elif p_j == self.piece:
                    features[x, y, min(j+1, 5)-2] += 1
                if p_i == -self.piece and p_j == -self.piece:
                    features[x, y, min(i+j+1, 5)+2] += 1
                elif p_i == -self.piece:
                    features[x, y, min(i+1, 5)+2] += 1
                elif p_j == -self.piece:
                    features[x, y, min(j+1, 5)+2] += 1
                ## up-down
                i, j = 0, 0
                p_i = game.board[x, y-1] if y-1 >= 0 else 0
                p_j = game.board[x, y+1] if y+1 < game.size else 0
                while p_i != 0 and y-i-1 >= 0 and game.board[x, y-i-1] == p_i: i += 1
                while p_j != 0 and y+j+1 < game.size and game.board[x, y+j+1] == p_j: j += 1
                if p_i == self.piece and p_j == self.piece:
                    features[x, y, min(i+j+1, 5)-2] += 1
                elif p_i == self.piece:
                    features[x, y, min(i+1, 5)-2] += 1
                elif p_j == self.piece:
                    features[x, y, min(j+1, 5)-2] += 1
                if p_i == -self.piece and p_j == -self.piece:
                    features[x, y, min(i+j+1, 5)+2] += 1
                elif p_i == -self.piece:
                    features[x, y, min(i+1, 5)+2] += 1
                elif p_j == -self.piece:
                    features[x, y, min(j+1, 5)+2] += 1
                ## NW-SE
                i, j = 0, 0
                p_i = game.board[x-1, y-1] if (x-1 >= 0 and y-1 >= 0) else 0
                p_j = game.board[x+1, y+1] if (x+1 < game.size and y+1 < game.size) else 0
                while p_i != 0 and x-i-1 >= 0 and y-i-1 >= 0 and game.board[x-i-1, y-i-1] == p_i: i += 1
                while p_j != 0 and x+j+1 < game.size and y+j+1 < game.size and game.board[x+j+1, y+j+1] == p_j: j += 1
                if p_i == self.piece and p_j == self.piece:
                    features[x, y, min(i+j+1, 5)-2] += 1
                elif p_i == self.piece:
                    features[x, y, min(i+1, 5)-2] += 1
                elif p_j == self.piece:
                    features[x, y, min(j+1, 5)-2] += 1
                if p_i == -self.piece and p_j == -self.piece:
                    features[x, y, min(i+j+1, 5)+2] += 1
                elif p_i == -self.piece:
                    features[x, y, min(i+1, 5)+2] += 1
                elif p_j == -self.piece:
                    features[x, y, min(j+1, 5)+2] += 1
                ## NE-SW
                i, j = 0, 0
                p_i = game.board[x+1, y-1] if (x+1 < game.size and y-1 >= 0) else 0
                p_j = game.board[x-1, y+1] if (x-1 >= 0 and y+1 < game.size) else 0
                while p_i != 0 and x+i+1 < game.size and y-i-1 >= 0 and game.board[x+i+1, y-i-1] == p_i: i += 1
                while p_j != 0 and x-j-1 >= 0 and y+j+1 < game.size and game.board[x-j-1, y+j+1] == p_j: j += 1
                if p_i == self.piece and p_j == self.piece:
                    features[x, y, min(i+j+1, 5)-2] += 1
                elif p_i == self.piece:
                    features[x, y, min(i+1, 5)-2] += 1
                elif p_j == self.piece:
                    features[x, y, min(j+1, 5)-2] += 1
                if p_i == -self.piece and p_j == -self.piece:
                    features[x, y, min(i+j+1, 5)+2] += 1
                elif p_i == -self.piece:
                    features[x, y, min(i+1, 5)+2] += 1
                elif p_j == -self.piece:
                    features[x, y, min(j+1, 5)+2] += 1
        return features

                
                