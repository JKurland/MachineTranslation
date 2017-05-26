from rnns import *
from copy import copy
from wrappers import *

def create_model(features, labels, params, mode):
    """creates the model, this means creating all variables and model edges
    
    Args:
        features: a dictionary of features, this will be converted to a list for convenience so the
                  keys should all be integers, like list indices
        labels: a dictionary of labels, will also be converted to a list
        params: dictionary of parameters, the parameters are:
            - buckets: the sizes of buckets into which samples will be put, in the form (source, target)
            - keep_prob: keep probability for dropout layers
            - alpha: initial value for the learning rate
            - layer_size: the size of each of the RNN layers (same for encoder and decoder)
            - num_layers: number of RNN cells to be stacked (same for encoder and decoder)
            - input_symbols: number of symbols in the source language
            - output_symbols: number of symbols in the target language
            - embedding_size: the size of the word embedding, (same for encoder and decoder)
            - attn_method: the method used for computing attention weights, one of:
                # 'lin': linear function of hidden layer and the state being attented
                # 'quad': almost general quadratic of hidden layer and encoder state. Terms which
                          aren't included are internal cross terms, that is a product of hidden
                          layer node with another hidden layer node, and the same for encoder state
                          terms
                # 'dot': the dot product of the hidden layer and encoder state, encoder state and 
                         hidden layer must be the same size
                # 'nn': a single hidden layer neural network
            - attn_layer_size: if the attn_method parameter is 'nn', sets the size of the nn hidden layer
            - max_grad_norm: the maximum allowed 2 norm of the gradients, gradients with a 2 norm larger
                             than this are clipped down
            - softmax_sample_size: number of samples to take when estimating the softmax error
        
        mode: the mode in which the model is being run, TRAIN, INFER or EVAL
        
    
    """
    
    
    layer_size = params['layer_size']
    num_layers = params['num_layers']
    input_symbols = params['input_symbols']
    output_symbols = params['output_symbols']
    embedding_size = params['embedding_size']
    attn_layer_size = params['attn_layer_size']
    max_grad_norm = params['max_grad_norm']
    softmax_sample_size = params['softmax_sample_size']
    is_training = params['is_training']
    buckets = params['buckets']
    keep_prob = params['keep_prob']
    attn_method = params['attn_method']
    
    #create the encoder cells, one for each bucket, all sharing weigths
    encoder_cells = []
    for i,bucket in enumerate(buckets):
        #core LSTM cell
        encoder_cells.append(tf.contrib.rnn.BasicLSTMCell(layer_size, reuse = True if i>0 else None))
        
        #stack the LSTMs
        encoder_cells[-1] = tf.contrib.rnn.MultiRNNCell([DropoutWrapper(copy(encoder_cells[-1]), keep_prob) for _ in range(num_layers)])
        
        #Add an embedding wrapper
        encoder_cells[-1] = EmbeddingWrapper(encoder_cells[-1], input_symbols, embedding_size, reuse = True if i>0 else None)
        
    #create decoder cells, one for each bucket, all sharing weights
    decoder_cells = []
    for i,bucket in enumerate(buckets):       
        #core LSTM cell
        decoder_cells.append(tf.contrib.rnn.BasicLSTMCell(layer_size, reuse = True if i>0 else None))

        #add one of three attention methods, the first is implemented as a wrapper, the second two
        #as a new cell. Then stack the cells as per the attention method
        
        #task 1, as in https://arxiv.org/pdf/1409.0473.pdf 
        if params['task'] == 1:
            decoder_cells[-1] = tf.contrib.rnn.MultiRNNCell([DropoutWrapper(copy(decoder_cells[-1]),keep_prob) for _ in range(num_layers)])
            decoder_cells[-1] = AttentionWrapper(decoder_cells[-1], [bucket[0], layer_size], attn_layer_size, state_is_tuple = True, reuse = True if i>0 else None)

        #task 2, as in http://aclweb.org/anthology/D15-1166
        if params['task'] == 2:
            attn_cell = AttentionCell(attn_method, layer_size, [bucket[0], layer_size], nn_hidden_size = 20, reuse = True if i>0 else None, project_output = True, projection_size = layer_size)
            decoder_cells[-1] = tf.contrib.rnn.MultiRNNCell( [DropoutWrapper(copy(decoder_cells[-1]),keep_prob) for _ in range(num_layers)] + [attn_cell])
        
        #like in http://aclweb.org/anthology/D15-1166 but with the attention layer within the rnn stack, the concatenated output is projected down to the correct size
        if params['task'] == 3:
            attn_cell = AttentionCell(attn_method, layer_size, [bucket[0], layer_size], nn_hidden_size = 20, reuse = True if i>0 else None, project_output = True, projection_size = layer_size)
            decoder_cells[-1] = tf.contrib.rnn.MultiRNNCell([DropoutWrapper(copy(decoder_cells[-1]),keep_prob)] + [attn_cell] + [DropoutWrapper(copy(decoder_cells[-1]),keep_prob) for _ in range(num_layers-1)])
    
        #add an embedding wrapper
        decoder_cells[-1] = EmbeddingWrapper(decoder_cells[-1], output_symbols, embedding_size, reuse = True if i>0 else None)
        
    params['decoder_output_size'] = decoder_cells[-1].output_size
    return create_model_from_cells(features, labels, params, encoder_cells, decoder_cells, mode)







def create_model_from_cells(features, labels, params, encoder_cells, decoder_cells, mode):
    """ creates a seq2seq model for each bucket using the encoder and decoder cells
    
    Args:
        features: a dict containing the input to the encoder and decoder layers
                must be a dictionary with integer keys (i.e. {0:..., 1:..., 2:...})
                as it will be converted into an ordered list. The inputs for all of
                the encoder buckets should come first, followed by the decoder buckets
        labels: labels must be a numpy array of shape (batch_size*bucket[1], 1)
                each element refers to the index of the correct symbol
        params: dict of parameters, these are unpacked at the top of the
                function (below)
        encoder_cells: the cells for the encoder RNN, one for each bucket
        decoder_cells: the cells for the decoder RNN, one for each bucket
    Raises:
        ValueError: if there are not two features for each bucket (one for encoder
                    one for decoder)
            
    
    """
    
    
    ##unpack params
    layer_size = params['layer_size']
    num_layers = params['num_layers']
    input_symbols = params['input_symbols']
    output_symbols = params['output_symbols']
    embedding_size = params['embedding_size']
    attn_layer_size = params['attn_layer_size']
    max_grad_norm = params['max_grad_norm']
    softmax_sample_size = params['softmax_sample_size']
    is_training = params['is_training']
    buckets = params['buckets']
    decoder_output_size = params['decoder_output_size']
    
    
    num_buckets = len(buckets)
    features = [features[i] for i in range(len(features))] #convert features from a dictionary to a list
    
    


    
    if type(features[0]) is tf.Tensor:
        feature_shapes = [x.get_shape().as_list() for x in features]
    else:
        feature_shapes = [x.shape for x in features]
    
    
        
    if not len(features) == len(buckets)*2: #encoder and decoder inputs for each bucket
        raise ValueError("Expected {} features, got {}".format(len(buckets*2), len(features)))

    ## encoder
    #compute the encoder sequence mask and lengths, any PAD tokens are considered out of sequence
    #everything else is in sequence
    enc_seq_masks = []
    enc_seq_lens = []
    for i in range(num_buckets):
        enc_seq_masks.append(tf.cast(tf.not_equal(features[i], tf.zeros_like(features[i])+PAD, name = 'test'), tf.int32))
        enc_seq_lens.append(tf.reshape(tf.reduce_sum(enc_seq_masks[-1], 1), [-1]))
    
    

        
        
    ## decoder
    #compute the decoder sequence mask and lengths
    dec_seq_masks = []
    dec_seq_lens = []
    for i in range(num_buckets):
        dec_seq_masks.append(tf.cast(tf.not_equal(features[i+num_buckets], tf.zeros_like(features[i+num_buckets])+PAD), tf.int32))
        dec_seq_lens.append(tf.reshape(tf.reduce_sum(dec_seq_masks[-1], 1), [-1]))
        
    #define the output projection
    with tf.variable_scope('output_projection'):
        w = tf.get_variable('output_w', [output_symbols, decoder_output_size])
        b = tf.get_variable('output_b', [output_symbols])
        
    
    loss = []
    prediction = []
    token_prediction = []
    num_non_zero = []
    ## seq2seq rnn
    for i, bucket in enumerate(buckets):
        #makes the actual RNN
        prediction.append(MyRNN(encoder_cells[i],
                                decoder_cells[i],
                                (w,b), 
                                features[i],
                                features[i+num_buckets],
                                is_training,
                                enc_seq_lens[i],
                                dec_seq_lens[i],
                                bucket,
                                layer_size))
    
        prediction[-1].set_shape([None, bucket[1], decoder_output_size])
        
        #deal with losses if mode is INFER
        if mode == tf.contrib.learn.ModeKeys.INFER:
            loss.append(0)
            num_non_zero.append(1)
            
            #make prediction of token
            projection = tf.einsum('ijk,lk -> ijl', prediction[-1], w) + b
            token_prediction.append(tf.argmax(tf.nn.softmax( projection ), axis = 2))
        #calculate real losses
        else:
            num_non_zero.append(tf.reduce_sum(dec_seq_lens[i]))
            
            l = tf.nn.sampled_softmax_loss(w, b,
                                            tf.convert_to_tensor(labels[i]),
                                            tf.reshape(prediction[-1], [-1, decoder_output_size]),
                                            softmax_sample_size,
                                            output_symbols,
                                            partition_strategy = 'div')
            #ignore the loss for out of sequence tokens, this keeps the choice of bucket size
            #from effecting the training
            l = tf.reduce_sum( l * tf.cast(tf.reshape(dec_seq_masks[i], [-1]), tf.float32))
            loss.append(l)
            token_prediction.append(tf.zeros([0]))
        
    return token_prediction, loss, num_non_zero