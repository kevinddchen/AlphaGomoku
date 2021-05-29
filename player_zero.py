import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import gomoku

def net(size, l2=1e-2):
    ''' Neural network that computes policy and value. '''
    input_layer = keras.Input(shape=(size, size, 1), name='input')
    x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='conv1')(input_layer)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='conv2')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='conv3')(x)
    ## policy head
    y = keras.layers.Conv2D(filters=2, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='pre_policy')(x)
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(size*size, activation='softmax', kernel_regularizer=keras.regularizers.L2(l2), name='policy')(y)
    ## value head
    x = keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='pre_value')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1, activation='tanh', kernel_regularizer=keras.regularizers.L2(l2), name='value')(x)
    
    return keras.Model(input_layer, [y, x])
    

    
class ZeroPlayer(gomoku.Player):
    '''Move according to AlphaGo Zero algorithm.
    
    Each time 'play()' is called, a game tree with 400 moves is generated using 'expand()'. Each 'expand()' down the 
    tree picks moves guided by the NN model. These 400 moves form the policy of the current state, and a move
    is finally selected. At each move, the (board, policy, value) are cached for training the NN model.
    
    Parameters =========

        name: string. Name of player.
        piece: int. Either +1 for black or -1 for white.
        game: Gomoku. Instance of new game.
        model: keras.Model. NN model that guides 'expand()'.
        recorder: GameRecorder. Object that caches data for NN training. Defaults to 'None', i.e. data not cached.

    Methods ============

        play(game: Gomoku, n_iter: int) -> (x, y)
            Returns move to play. 'n_iter' is number of expand() to generate in the game tree. Defaults to 400.
            
        update(move: int)
            Update the game tree by one move, where 'move' = x*size + y.
            
        expand()
            Traverse down game tree and add one node.
    '''
    
    def __init__(self, name, piece, game, model, recorder=None):
        super().__init__(name, piece)
        self.tree = MCTree(piece, game, model)
        self.model = model
        self.recorder = recorder
    
    def play(self, game, n_iter=1000):
        ## if needed, update tree with most recent move
        if self.tree.n_moves != len(game.episode):
            x, y = game.episode[-1]
            self.tree.updateHead(x*game.size + y)
        ## do MC tree search
        for i in range(n_iter):
            node = self.tree.select()
            self.tree.backup(node)
        ## compute policy
        policy, value = self.tree.policy_value()
        ## save cache
        if self.recorder:
            self.recorder.write(game.board * self.piece, policy, value)
        ## pick move according to policy
        move = np.random.choice(game.size*game.size, p=policy)
        self.tree.updateHead(move)
        x, y = move//game.size, move%game.size
        return x, y
        
        
        
class MCTree:
    ''' Data structure to handle game tree.'''
    
    def __init__(self, piece, game, model):
        self.piece = piece
        self.model = model
        self.head = MCTreeNode(game, None)
        self.evaluate(self.head)
        self.prev_head = None # useful for debugging
        self.n_moves = 0
        self.laplace = 1. # Laplace smoothing
        
    def pickMove(self, node):
        ''' Pick move using PUCB algorithm. '''
        assert not node.game.finished, "game is finished"
        Q = node.Q.copy()
        Q[node.N == 0] = 1
        C = np.zeros_like(Q)
        if node.t > 0:
            mask = node.N != 0
            C[mask] = np.sqrt(3*np.log(node.t) / (2*node.N + 1e-6))[mask]
        M = 2 / (node.P + 1e-6)
        if node.t > 1:
            M *= np.sqrt( np.log(node.t)/node.t )
        UCB = Q + C - M
        ## forbid moves
        UCB[node.forbid] = np.min(UCB) - 1
        return np.argmax(UCB)
        
    def select(self):
        ''' Traverse down game tree and return a leaf node, expanding if needed. '''
        node = self.head
        while not node.game.finished:
            move = self.pickMove(node)
            if move not in node.children: # move has not been explored
                return self.expand(node, move)
            node = node.children[move]
        return node # node is terminal state
    
    def expand(self, parent, move):
        ''' Add node to game tree and evaluate using NN model. '''
        child = MCTreeNode(parent.game, parent)
        parent.children[move] = child
        child.parent_id = move
        ## update game state of child
        x, y = move//parent.game.size, move%parent.game.size
        child.game.play(x, y)
        ## evaluate child
        self.evaluate(child)
        return child
    
    def evaluate(self, node):
        ''' If game is terminal, assign V = +/- 1. Else, evaluate using NN model. '''
        if node.game.finished:
            node.V = node.game.winner * self.piece
        else:
            tensor = node.game.board.reshape((1, node.game.size, node.game.size, 1)) * node.game.curr_player
            P, V = self.model(tensor)
            node.P = P.numpy()[0]
            node.V = V.numpy()[0, 0]
            node.forbid = node.game.forbidden_actions_list()
            
    def backup(self, node):
        while node.parent:
            V = node.V
            move = node.parent_id
            node = node.parent
            node.t += 1
            node.N[move] += 1
            node.Q[move] += (V - node.Q[move])/node.N[move]
    
    def updateHead(self, move):
        self.n_moves += 1
        self.prev_head = self.head
        if move not in self.head.children: # move has not been explored
            self.head = MCTreeNode(self.head.game, None)
            x, y = move//self.head.game.size, move%self.head.game.size
            self.head.game.play(x, y)
            self.evaluate(self.head)
        else: # move has been explored
            self.head = self.head.children[move]
            self.head.parent = None
    
    def policy_value(self):
        policy = self.head.N + self.laplace * self.head.game.available_actions().flatten()
        policy /= np.sum(policy)
        return policy, np.sum(policy * self.head.Q)
        
    
    
class MCTreeNode:
    
    def __init__(self, game, parent):
        self.game = game.copy()
        self.parent = parent
        self.parent_id = None
        self.children = {}
        self.P = None
        self.V = None
        self.forbid = None
        self.Q = np.zeros(game.size**2, dtype=np.float32)
        self.N = np.zeros(game.size**2, dtype=np.float32)
        self.t = 0
        


        
        
class GameRecorder:
    ''' Custom object to read/write game data as TFRecords file.
    
    To record data:
        recorder = GameRecorder([filename goes here])
        recorder.open()
        [ run games here ]
        recorder.close()
    
    To read data:
        recorder = GameRecorder([filename goes here])
        data = recorder.fetch()
    '''
    
    def __init__(self, filename):
        self.filename = filename
        self.feature_description = {
            'board': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'policy': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'value': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
            }
    
    def _parse_function(self, example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary `feature_description`.
        dct = tf.io.parse_single_example(example_proto, self.feature_description)
        board = tf.reshape(tf.io.decode_raw(dct['board'], out_type=tf.int8), (9, 9, 1))
        policy = tf.reshape(tf.io.decode_raw(dct['policy'], out_type=tf.float32), (81,))
        return (board, {'policy': policy, 'value': dct['value']})
    
    ## ============================================================================
    ## The following functions are adapted from the TensorFlow tutorial,
    ## https://www.tensorflow.org/tutorials/load_data/tfrecord
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    ## ============================================================================
        
    def open(self, overwrite=False):
        assert overwrite or not os.path.exists(self.filename), "file already exists"
        self.writer = tf.io.TFRecordWriter(self.filename)
        
    def write(self, board, policy, value):
        feature = {
            'board': self._bytes_feature(board.tobytes()),
            'policy': self._bytes_feature(policy.tobytes()),
            'value': self._float_feature(value)
            }
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        self.writer.write(tf_example.SerializeToString())
        
    def close(self):
        self.writer.close()
    
    def fetch(self):
        return tf.data.TFRecordDataset(self.filename).map(self._parse_function)
    

    


    