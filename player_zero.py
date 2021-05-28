import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import gomoku

def net(size, l2=1e-4):
    ''' Neural network for computing policy and value. '''
    input_layer = keras.Input(shape=(size, size, 1), name='input')
    x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='conv1')(input_layer)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='conv2')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='conv3')(x)
    ## policy head
    y = keras.layers.Conv2D(filters=2, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='policy_conv')(x)
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(size*size, activation='softmax', kernel_regularizer=keras.regularizers.L2(l2), name='policy')(y)
    ## value head
    x = keras.layers.Conv2D(filters=1, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='value_conv')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.L2(l2), name='value_dense')(x)
    x = keras.layers.Dense(1, activation='tanh', kernel_regularizer=keras.regularizers.L2(l2), name='value')(x)
    
    return keras.Model(input_layer, [y, x])
    

    
class ZeroPlayer(gomoku.Player):
    def __init__(self, name, piece, game, model, recorder=None):
        super().__init__(name, piece)
        self.tree = MCTreeNode(game.copy())
        self.tree.evaluate(model, piece)
        self.model = model
        self.recorder = recorder
    
    def play(self, game, n_iter=400):
        ## update head to current game state
        while len(self.tree.game.episode) < len(game.episode):
            x, y = game.episode[len(self.tree.game.episode) - len(game.episode)]
            self.update(x*game.size + y)
        ## do MC tree search
        for i in range(n_iter):
            self.expand()
        ## compute policy
        policy = self.tree.N
        policy /= np.sum(policy)
        ## save cache
        if self.recorder:
            value = np.sum(policy * self.tree.Q)
            self.recorder.write(game.board * self.piece, policy, value)
        ## pick move according to policy
        move = np.random.choice(game.size*game.size, p=policy)
        self.update(move)
        x, y = move//game.size, move%game.size
        return x, y
    
    def update(self, move):
        ''' Update tree by one move. '''
        ## move has not been explored
        if not self.tree.children[move]:
            self.tree = MCTreeNode(self.tree.game.copy())
            self.tree.makeMove(move)
            self.tree.evaluate(self.model, self.piece)
        ## move has been explored
        else:
            self.tree = self.tree.children[move]
            self.tree.parent = None
            
    def expand(self):
        ''' Traverse down game tree according to PUCT and add a new leaf node. '''
        ## traverse down
        node = self.tree
        while not node.game.finished:
            move = node.pickMove()
            #print(move, end=' ')
            if not node.children[move]:
                leaf = MCTreeNode(node.game.copy(), node)
                leaf.makeMove(move)
                leaf.evaluate(self.model, self.piece)
                node.addChild(move, leaf)
                node = leaf
                break
            node = node.children[move]
        ## traverse up
        while node.parent:
            V = node.V
            move = node.move
            node = node.parent
            node.N[move] += 1
            node.Q[move] += (V - node.Q[move])/node.N[move]
        
                

class MCTreeNode:
    
    def __init__(self, game, parent=None):
        N = game.size**2
        self.game = game
        self.parent = parent
        self.children = [None for i in range(N)]
        self.move = None
        self.P = None
        self.V = None
        self.forbid = None
        self.Q = np.zeros(N, dtype=np.float32)
        self.N = np.zeros(N, dtype=np.float32)
        
    def makeMove(self, move):
        ''' Make a move on node's game. ONLY DO THIS ONCE! '''
        x, y = move//self.game.size, move%self.game.size
        self.game.play(x, y)
        self.move = move
        
    def evaluate(self, model, piece):
        ''' Evaluate node's game. If terminal, only V is defined. Else, use NN. '''
        if self.game.finished:
            self.V = self.game.winner * piece
        else:
            tensor = self.game.board.reshape((1, self.game.size, self.game.size, 1)) * self.game.curr_player
            P, V = model(tensor)
            self.P = P.numpy()[0]
            self.V = V.numpy()[0, 0]
            self.forbid = self.game.forbidden_actions_list()
        
    def pickMove(self, c=2.5):
        ''' Pick move using PUCT. '''
        assert not self.game.finished, "game is finished"
        UCB = self.Q + c * np.sqrt(1+np.sum(self.N)) * self.P / (1+self.N)
        ## forbid moves
        UCB[self.forbid] = np.min(UCB) - 1
        actions = np.where(UCB == np.max(UCB))[0]
        return np.random.choice(actions)
    
    def addChild(self, move, child):
        self.children[move] = child
        child.parent = self

        
        
class GameRecorder:
    
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
    

    


    