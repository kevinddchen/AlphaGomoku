import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import gomoku

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def net(size, l2=1e-4):
    ''' Neural network f_\theta that computes policy and value. '''
    input_layer = keras.Input(shape=(size, size, 4), name='input')
    x = keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='conv1')(input_layer)
    x = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='conv2')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='conv3')(x)
    ## policy head
    y = keras.layers.Conv2D(filters=4, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='conv4')(x)
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(size*size, activation='softmax', kernel_regularizer=keras.regularizers.L2(l2), name='policy')(y)
    ## value head
    x = keras.layers.Conv2D(filters=2, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', 
                            kernel_regularizer=keras.regularizers.L2(l2), name='conv5')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.L2(l2), name='dense1')(x)
    x = keras.layers.Dense(1, activation='tanh', kernel_regularizer=keras.regularizers.L2(l2), name='value')(x)
    
    return keras.Model(input_layer, [y, x])
    

    
class ZeroPlayer(gomoku.Player):
    '''Move according to AlphaGo Zero algorithm.
    
    Each time 'play()' is called, the game tree is simulated n times using 'select()'. Each 'select()' down 
    the tree picks moves guided by a UCB multi-arm bandit algorithm, until it reaches a terminal or previously 
    unvisited state. In the latter case, the node is added to the tree with 'expand()' and the prior policy and 
    value are esimated using 'evaluate()'. 'backup()' propagates the results back up the game tree.
    
    After the MCTS simulation, a move is selected from a policy \pi formed by the simulations, according to
    'get_policy_value()'. For each played move, the (board, policy, value) can be cached for training the NN model.
    
    Uses four features:
        0) 1 for current player's stones, 0 otherwise
        1) 1 for opponent's stones, 0 otherwise
        2) 1 for opponent's most recent move, 0 otherwise
        3) all 1 if current player is black, all 0 if current player is white
    
    Parameters =========

        name <string> ... Name of player.
        piece <int> ... Either +1 for black or -1 for white.
        game <Gomoku> ... Instance of new game.
        model <keras.Model> ... NN model that evaluates new states.
        recorder <GameRecorder> ... Object that caches data for NN training. Defaults to 'None', i.e. data not cached.
        
    Variables ============
    
        name <string>
        piece <int>
        tree <MCTree> ... Game tree.
        model <keras.Model>
        recorder <GameRecorder> 

    Methods ============

        play(game, n_iter) ... Given Gomoku game, returns move (x, y) to play. 
            'n_iter' is an integer number of 'expand()' to generate in the game tree. 
    '''
    
    def __init__(self, name, piece, game, model, recorder=None):
        super().__init__(name, piece)
        self.tree = MCTree(game, model)
        self.model = model
        self.recorder = recorder
    
    def play(self, game, n_iter):
        ## if needed, update game tree with opponent's most recent move
        if self.tree.n_moves != len(game.episode):
            x, y = game.episode[-1]
            self.tree.updateHead(x*game.size + y)
        ## do MC tree search
        for i in range(n_iter):
            node = self.tree.select()
            self.tree.backup(node)
        ## compute policy and value from the MCTS results
        policy, value = self.tree.get_policy_value()
        ## save cache
        if self.recorder:
            tensor = self.tree.boardToTensor(game.board, self.piece, self.tree.head.parent_id)
            self.recorder.write(tensor, policy, value)
        ## pick move according to policy
        move = np.random.choice(len(policy), p=policy)
        self.tree.updateHead(move)
        x, y = move//game.size, move%game.size
        return x, y
        
        
        
class MCTree:
    ''' Data structure to handle game tree.'''
    
    def __init__(self, game, model):
        ## hyperparameters ==========
        self.c = 4. # controls UCB exploration
        self.dirichlet = .3 # controls dirichlet noise
        self.epsilon = .25 # controls amount of dirichlet noise to add
        self.temp = .1 # controls exploration of output policy
        ## ==========================
        self.model = model
        self.head = MCTreeNode(game, None)
        self.prev_head = None # useful for debugging
        self.n_moves = len(game.episode) # number of moves played on board (not number of tree simulations)
        self.evaluate(self.head)
        
    def pickMove(self, node):
        ''' Pick a move using UCB multi-arm bandit algorithm. '''
        #assert not node.game.finished, "game is finished"
        UCB = node.Q + self.c * node.P * np.sqrt(max(1, node.t)) / (1+node.N) # UCB = Q + U defined for AlphaGo Zero
        UCB[node.game.forbidden_actions_list()] = np.min(UCB) - 1 # forbid moves
        return np.argmax(UCB)
        
    def select(self):
        ''' Traverse down game tree and return a leaf node, expanding if needed. '''
        node = self.head
        while not node.game.finished:
            move = self.pickMove(node)
            if move not in node.children: # node has not been explored
                return self.expand(node, move)
            node = node.children[move]
        return node # node is terminal state
    
    def expand(self, parent, move):
        ''' Add node to game tree and evaluate it. '''
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
        ''' Evaluate using NN model. '''
        if node.game.finished: # game is terminal
            node.V = -1 # if it is your turn and you see a finished board, you have lost
        else: # game is not terminal
            tensor = self.boardToTensor(node.game.board, node.game.curr_player, node.parent_id)
            P, V = self.model(np.expand_dims(tensor, 0)) # n_samples=1 when passed to model
            node.V = V.numpy()[0, 0]
            node.P = P.numpy()[0]
            ## add dirichlet noise
            node.P = (1-self.epsilon)*node.P + self.epsilon*np.random.dirichlet(self.dirichlet*np.ones_like(node.P))
            
    def backup(self, node):
        ''' From leaf node, update Q and N of all parents nodes. '''
        V = node.V
        while node.parent:
            V *= -1 # flip value each player
            move = node.parent_id
            node = node.parent
            node.t += 1
            node.N[move] += 1
            if node.N[move] == 1: # if first time, remove optimism
                node.Q[move] = V 
            else:
                node.Q[move] += (V - node.Q[move])/node.N[move]
            
    def updateHead(self, move):
        ''' Update head by one move. '''
        self.n_moves += 1
        self.prev_head = self.head
        if move not in self.head.children: # move has not been explored
            self.head = MCTreeNode(self.head.game, None)
            self.head.parent_id = move
            x, y = move//self.head.game.size, move%self.head.game.size
            self.head.game.play(x, y)
            self.evaluate(self.head)
        else: # move has been explored
            self.head = self.head.children[move]
            self.head.parent = None
    
    def get_policy_value(self):
        policy = softmax(np.log(self.head.N + 1e-10) / self.temp).astype(np.float32)
        return policy, np.sum(policy * self.head.Q)
    
    def boardToTensor(self, board, piece, prev_move):
        ''' Returns shape (size, size, 4) tensor '''
        size = len(board)
        tensor = np.zeros((size, size, 4), dtype=np.int8)
        tensor[board==piece, 0] = 1
        tensor[board==-(piece), 1] = 1
        if prev_move != None:
            x, y = prev_move//size, prev_move%size
            tensor[x, y, 2] = 1
        if piece == 1:
            tensor[..., 3] = 1
        return tensor
        
    
    
class MCTreeNode:
    
    def __init__(self, game, parent):
        self.game = game.copy()
        self.parent = parent
        self.parent_id = None # self = parent.children[parent_id]
        self.children = {}
        self.P = None
        self.V = None
        self.Q = np.ones(game.size**2, dtype=np.float32) # optimism (initialize to 1 to visit every action)
        self.N = np.zeros(game.size**2, dtype=np.float32)
        self.t = 0 # t = sum(N)
      
        
        
class GameRecorder:
    ''' Custom object to read/write game data as TFRecords file.
    
    To record data:
        recorder = GameRecorder("name_of_file.tfrecords")
        recorder.open()
        [ run games here ]
        recorder.close()
    
    To read data:
        recorder = GameRecorder("name_of_file.tfrecords")
        data = recorder.fetch()
        
    'open()' will not to overwrite existing files. This can be overridden by 'GameRecorder(..., overwrite=True)'.
    '''
    
    def __init__(self, filename, size, overwrite=False):
        self.size = size
        self.filename = filename
        self.overwrite = overwrite
        self.feature_description = {
            'board': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'policy': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'value': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
            }
        
    def __enter__(self, **kwargs):
        self.open(**kwargs)
        return self
   
    def __exit__(self, exception_type, exception_value, traceback):
        self.writer.close()
    
    def _parse_function(self, example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary `feature_description`.
        dct = tf.io.parse_single_example(example_proto, self.feature_description)
        board = tf.reshape(tf.io.decode_raw(dct['board'], out_type=tf.int8), (self.size, self.size, 4))
        policy = tf.reshape(tf.io.decode_raw(dct['policy'], out_type=tf.float32), (self.size**2,))
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
        
    def open(self):
        assert self.overwrite or not os.path.exists(self.filename), "file already exists"
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
        assert os.path.exists(self.filename), "file does not exist"
        return tf.data.TFRecordDataset(self.filename).map(self._parse_function)
    
    

class PrintRecorder:
    ''' Does not cache data; just prints it'''
    def write(self, board, policy, value):
        print("board[0]:\n", board[..., 0])
        print("board[1]:\n", board[..., 1])
        print("board[2]:\n", board[..., 2])
        print("board[3]:\n", board[..., 3])
        print("policy:\n", policy)
        print("value:", value)
          

    