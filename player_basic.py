import numpy as np
import gomoku

class RandomPlayer(gomoku.Player):
    '''Makes random moves.'''
    def __init__(self, name, piece):
        super().__init__(name, piece)

    def play(self, game):
        avail_acts = game.available_actions_list()
        move = np.random.choice(avail_acts)
        return move//game.size, move%game.size
    

    
class FeaturePlayer(gomoku.Player):
    '''Move according to basic features.
    
        get_features() returns a [size, size, 9]-shape array containing 9 features for each position of the board:
            0: Number of 2-chains the move will create.
            1: "       " 3-chains "                  ".
            2: "       " 4-chains "                  ".
            3: "       " 5-chains "                  ".
            4: Number of opponent 2-chains the move will block.
            5: "                " 3-chains "                 ".
            6: "                " 4-chains "                 ".
            7: "                " 5-chains "                 ".
            8: Ones (for bias).
        The w vector assigns points to each feature, and adds them up. Player makes move with maximum points.
    
    Variables ===========
    
        name: string. Name of player.
        piece: int. Either +1 for black or -1 for white.
        w: array.
        
    Methods ============
    
        play(game: Gomoku) -> (x, y)
            Returns move to play.
    
        get_features(board: array, piece: int) -> array
            Returns array of features, described above.
    '''
    def __init__(self, name, piece):
        super().__init__(name, piece)
        ## default points assigned to each each feature
        self.w = np.array([1, 3, 9, 81, 1, 3, 9, 27, 0])
        
    def play(self, game):
        ## first move?
        if not game.episode:
            return (game.size-1)//2, (game.size-1)//2
        features = self.get_features(game.board, self.piece)
        scores = features.dot(self.w).flatten()
        ## mark forbidden moves
        forbid_acts = game.forbidden_actions_list()
        scores[forbid_acts] = np.min(scores) - 1
        ## pick move with max score
        actions = np.where(scores == np.max(scores))[0]
        move = np.random.choice(actions) # move = x*size + y
        return move//game.size, move%game.size
       
    def get_features(self, board, piece):
        ## calculate features as described above
        size = len(board)
        features = np.zeros((size, size, 9))
        features[..., 8] = 1
        for x in range(size):
            for y in range(size):
                ## left-right
                i, j = 0, 0
                p_i = board[x-1, y] if x-1 >= 0 else 0
                p_j = board[x+1, y] if x+1 < size else 0
                while p_i != 0 and x-i-1 >= 0 and board[x-i-1, y] == p_i: i += 1
                while p_j != 0 and x+j+1 < size and board[x+j+1, y] == p_j: j += 1
                if p_i == piece and p_j == piece:
                    features[x, y, min(i+j+1, 5)-2] += 1
                elif p_i == piece:
                    features[x, y, min(i+1, 5)-2] += 1
                elif p_j == piece:
                    features[x, y, min(j+1, 5)-2] += 1
                if p_i == -piece and p_j == -piece:
                    features[x, y, min(i+j+1, 5)+2] += 1
                elif p_i == -piece:
                    features[x, y, min(i+1, 5)+2] += 1
                elif p_j == -piece:
                    features[x, y, min(j+1, 5)+2] += 1
                ## up-down
                i, j = 0, 0
                p_i = board[x, y-1] if y-1 >= 0 else 0
                p_j = board[x, y+1] if y+1 < size else 0
                while p_i != 0 and y-i-1 >= 0 and board[x, y-i-1] == p_i: i += 1
                while p_j != 0 and y+j+1 < size and board[x, y+j+1] == p_j: j += 1
                if p_i == piece and p_j == piece:
                    features[x, y, min(i+j+1, 5)-2] += 1
                elif p_i == piece:
                    features[x, y, min(i+1, 5)-2] += 1
                elif p_j == piece:
                    features[x, y, min(j+1, 5)-2] += 1
                if p_i == -piece and p_j == -piece:
                    features[x, y, min(i+j+1, 5)+2] += 1
                elif p_i == -piece:
                    features[x, y, min(i+1, 5)+2] += 1
                elif p_j == -piece:
                    features[x, y, min(j+1, 5)+2] += 1
                ## NW-SE
                i, j = 0, 0
                p_i = board[x-1, y-1] if (x-1 >= 0 and y-1 >= 0) else 0
                p_j = board[x+1, y+1] if (x+1 < size and y+1 < size) else 0
                while p_i != 0 and x-i-1 >= 0 and y-i-1 >= 0 and board[x-i-1, y-i-1] == p_i: i += 1
                while p_j != 0 and x+j+1 < size and y+j+1 < size and board[x+j+1, y+j+1] == p_j: j += 1
                if p_i == piece and p_j == piece:
                    features[x, y, min(i+j+1, 5)-2] += 1
                elif p_i == piece:
                    features[x, y, min(i+1, 5)-2] += 1
                elif p_j == piece:
                    features[x, y, min(j+1, 5)-2] += 1
                if p_i == -piece and p_j == -piece:
                    features[x, y, min(i+j+1, 5)+2] += 1
                elif p_i == -piece:
                    features[x, y, min(i+1, 5)+2] += 1
                elif p_j == -piece:
                    features[x, y, min(j+1, 5)+2] += 1
                ## NE-SW
                i, j = 0, 0
                p_i = board[x+1, y-1] if (x+1 < size and y-1 >= 0) else 0
                p_j = board[x-1, y+1] if (x-1 >= 0 and y+1 < size) else 0
                while p_i != 0 and x+i+1 < size and y-i-1 >= 0 and board[x+i+1, y-i-1] == p_i: i += 1
                while p_j != 0 and x-j-1 >= 0 and y+j+1 < size and board[x-j-1, y+j+1] == p_j: j += 1
                if p_i == piece and p_j == piece:
                    features[x, y, min(i+j+1, 5)-2] += 1
                elif p_i == piece:
                    features[x, y, min(i+1, 5)-2] += 1
                elif p_j == piece:
                    features[x, y, min(j+1, 5)-2] += 1
                if p_i == -piece and p_j == -piece:
                    features[x, y, min(i+j+1, 5)+2] += 1
                elif p_i == -piece:
                    features[x, y, min(i+1, 5)+2] += 1
                elif p_j == -piece:
                    features[x, y, min(j+1, 5)+2] += 1
        return features
    
    
    
# class PGFeaturePlayer(FeaturePlayer):
#     '''Tries to learn by actor-critic policy gradient. Uses linear function 
#     approximation given by the features of Feature Player. Does not really work.'''
    
#     def __init__(self, name, piece, epsilon=.2, w_lr=.1, theta_lr=.1, discount=.8):
#         super().__init__(name, piece)
#         self.epsilon = epsilon
#         self.w_lr = w_lr
#         self.theta_lr = theta_lr
#         self.discount = discount
#         self.w = np.zeros(9)
#         self.theta = np.zeros(9)
#         self.cache = []
        
#     def play(self, game):
#         avail_moves = game.available_actions()
#         features = self.get_features(game)
#         probs = softmax(features @ self.theta)
#         probs[~avail_moves] = 0 # do not take invalid moves
#         probs = probs.flatten()/np.sum(probs) # re-normalize probabilities
#         ave_features = features.T.reshape(9, -1) @ probs
        
#         x = np.random.random()
#         ## play random
#         if x < self.epsilon:
#             #print("random")
#             move = np.random.choice(np.where(probs > 0)[0])
#         ## play according to policy
#         else:
#             #print("policy")
#             move = np.random.choice(len(probs), p=probs)
#         x, y = move//game.size, move%game.size
#         self.cache.append((x, y, features[x, y], ave_features)) 
#         return x, y
    
#     def get_grad(self, game, reward):
#         w_grad = np.zeros_like(self.w)
#         theta_grad = np.zeros_like(self.theta)
#         N = 0.
#         for x, y, phi, ave_phi in self.cache[::-1]:
#             w_grad += (reward - self.w @ phi)*phi
#             theta_grad += (self.w @ phi)*(phi - ave_phi)
#             N += 1
#             reward *= self.discount
#         self.cache = []
#         return w_grad/N, theta_grad/N
        
#     def step(self, w_grad, theta_grad):
#         self.w += self.w_lr * w_grad
#         self.theta += self.theta_lr * theta_grad
    
                
                