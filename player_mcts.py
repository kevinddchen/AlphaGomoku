import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import gomoku



def net(size):
    
    input_layer = keras.Input(shape=(size, size, 1), name='input')
    x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
                            padding='same', activation='relu', name='conv1')(input_layer)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
                            padding='same', activation='relu', name='conv2')(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
                            padding='same', activation='relu', name='conv3')(x)

    y = keras.layers.Conv2D(filters=1, kernel_size=(3,3), strides=(1,1),
                            padding='same', activation=None, name='pre_policy')(x)
    y = keras.layers.Flatten()(y)
    y = keras.layers.Activation(keras.activations.softmax, name='policy')(y)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1, activation='sigmoid', name='value')(x)
    
    return keras.Model(input_layer, [y, x])
    

    
class ZeroPlayer(gomoku.Player):
    def __init__(self, name, piece, game, model):
        super().__init__(name, piece)
        self.head = MCTreeNode(game)
        self.head.evaluate()
    
    def play(self, game):
        for i in range(5):
            node = self.expansion()
            self.backup(node)
            print()
        return (0, 0)
            
    def expansion(self):
        node = self.head
        while True:
            move = node.pickMove()
            print(move)
            if not node.children[move]:
                leaf = MCTreeNode(node.board, -node.piece, node)
                leaf.makeMove(move)
                leaf.evaluate()
                node.addChild(move, leaf)
                return leaf
            node = node.children[move]
    
    def backup(self, node):
        while node.parent:
            V = node.V
            move = node.move
            node = node.parent
            node.N[move] += 1
            node.Q[move] += (V - node.Q[move])/node.N[move]
                

class MCTreeNode:
    
    def __init__(self, game, parent=None):
        N = len(game.size)**2
        self.game = game
        self.parent = parent
        self.children = [None for i in range(N)]
        self.n_children = 0
        self.move = None
        self.P = None
        self.V = None
        self.Q = np.zeros(N)
        self.N = np.zeros(N)
        
    def makeMove(self, move):
        x, y = move//self.game.size, move%self.game.size
        self.game.play(x, y)
        self.move = move
        
    def evaluate(self, piece, model):
        if self.game.finished:
            if self.game.winner == piece:
                self.V = 1
            else:
                self.V = 0
        else:
            t = self.game.board.reshape((1, self.game.size, self.game.size, 1)) * self.game.curr_player
            P, V = model(t)
            self.P = P.numpy()[0]
            self.V = V.numpy()[0, 0]
            ## forbid moves
            forbid_acts = self.game.forbidden_actions_list()
            self.Q[forbid_acts] = -1
        
    def pickMove(self):
        UCB = self.Q + 1. * self.P / (1 + self.N)
        return np.argmax(UCB)

    def isLeaf(self):
        return self.n_children == 0
    
    def addChild(self, move, child):
        self.children[move] = child
        child.parent = self
        self.n_children += 1
    
    
# class MCTreeNode:
    
#     def __init__(self, board, piece, parent=None):
#         N = len(board)**2
#         self.board = np.copy(board)
#         self.piece = piece
#         self.parent = parent
#         self.children = [None for i in range(N)]
#         self.n_children = 0
#         self.move = None
#         self.P = None
#         self.V = None
#         self.Q = np.zeros(N)
#         self.N = np.zeros(N)
        
#     def makeMove(self, move):
#         size = len(self.board)
#         x, y = move//size, move%size
#         self.board[x, y] = self.piece
#         self.move = move
        
#     def evaluate(self):
#         forbid_acts = np.where((self.board != 0).flatten())[0]
#         self.P = np.random.rand(len(self.board)**2)
#         self.P[forbid_acts] = 0
#         self.P /= np.sum(self.P)
#         self.V = np.random.rand(1)/100.
        
#     def pickMove(self):
#         UCB = self.Q + 1. * self.P / (1 + self.N)
#         return np.argmax(UCB)

#     def isLeaf(self):
#         return self.n_children == 0
    
#     def addChild(self, move, child):
#         self.children[move] = child
#         child.parent = self
#         self.n_children += 1
    
    