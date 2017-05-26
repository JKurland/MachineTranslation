from tensorflow.python.util import nest
import tensorflow as tf
import numpy as np

PAD = 0
GO = 1
EOS = 2
UNK = 3

#cells form a directed graph, with each node also being a cell. Some cells have a single outgoing edge,
#these cells have an attribute _cell pointing to the next cell down. Other cells have mutliple
#outgoing edges, these cells have an attribute _cells, a list, pointing to each of the next cells.

#The state tuple has a similar structure, for cells with one outgoing edge, the state tuple is made of
#two parts ((next_cell_state), this_cell_state0, this_cell_state1 ...), the first is the state tuple
#of the next cell down, then the current cells states. For cells with mutliple outgoing edges the
#state is of the form ((next_cell1_state), (next_cell2_state), ...)

#The two functions below search the cell tree and state tree to find the positions of certain states.
#For example it is necessary to find cells which are implementing some form of attention, these cells
#need the encoder output fed to them as a state. The function find_attn_pos finds the position of the
#states that need the encoder output fed to them, the position is the index of the state when the state
#tuple is flattened.



def find_attn_pos(cell):
    """given a nested cell, find the positions of states which need the encoder output fed into them
    Args:
        cell: A cell to be searched, if no attention cells exist within cell an empty list is returned
    
    """
    
    if hasattr(cell, '_cell'):#if cell is a wrapper
        positions = find_attn_pos(cell._cell)
        if hasattr(cell, '_is_attn_cell'):
            positions.append(len(nest.flatten(cell._cell.state_size)))
            
        return positions
        
    elif hasattr(cell, '_cells'):#if cell is multiRNN
        positions = []
        l = 0
        for c in cell._cells:
            [positions.append(p + l) for p in find_attn_pos(c)]
            l = l+len(nest.flatten(c.state_size))
        
        return positions
        
    else: #if cell is a base cell
        positions = []
        if hasattr(cell, '_is_attn_cell'):
            positions.append(0)
            
        return positions

def find_hidden_pos(cell):
    """finds the state positions of base cells (e.g. LSTM, GRU... etc) so that these states can be
    passed from encoder to decoder. To detect these cells the following conditions are used
    1. the cell has no attribute _cell or _cells, this means that the cell is not a wrapper
    2. the cell has no attribute _do_not_find, this is a flag that can be added to custom cells
    
    Args:
        cell: the cell being searched
    
    returns: the positions of all the states which belong to base cell, positions are the index in
            the flat list
    """
    
    if hasattr(cell, '_cell'):#if cell is a wrapper
        positions = find_hidden_pos(cell._cell)
            
        return positions
        
    elif hasattr(cell, '_cells'):#if cell is multiRNN
        positions = []
        l = 0
        for c in cell._cells:
            [positions.append(p + l) for p in find_hidden_pos(c)]
            l = l+len(nest.flatten(c.state_size))
        
        return positions
        
    else: #if cell is a base cell
        positions = []
        if not hasattr(cell, '_do_not_find'):
            for i in range(len(nest.flatten(cell.state_size))):
                positions.append(i)
            
        return positions
    

#(finished, next_input, next_cell_state, emit_output, next_loop_state)

def MyRNN(encoder_cell, decoder_cell, output_proj, encoder_input, decoder_input, is_training,
          enc_sequence_length, dec_sequence_length, bucket, encoder_output_size, attn_state_positions=None,
          decoder_state_positions = None, encoder_state_positions = None):
    
    """RNN function which implements the seq2seq model from a decoder cell and an encoder cell.
    First dynamic_rnn is used to create the encoder from encoder cell, this produces the
    encoder outputs, an output at every time step, and the final encoder state tuple.
    The encoder output is fed into attention layers within the decoder, via their initial cell state.
    The final encoder state tuple is used to initialise the decoder states. First the encoder states
    which belong to the base cells (LSTM, GRU, etc...) are found, then the corresponding states in
    the decoder state tuple are found, there must be the same number of base cells in the encoder and
    decoder. The final state of the encoder cells are then fed into the decoder cells as intial states.
    
    During training the decoder is given the true, correct prevous symbol as input, that is, when
    translating the phrase "The dog" the decoder is first given the input GO, this should lead to
    the cell giving an output of "Le", the decoder then takes as input "Le" and should produce "chien".
    Then it takes "chien" as input and should produce the token for EOS. During training it does not
    matter what word the cell actually predicts, the decoder is always fed the correct translation,
    however, during inference and evaluation, the correct word is not available, so the decoder is
    fed its own prediction. Which of these methods to use is set by the argument is_training
    
    Args:
        encoder_cell: the RNN cell for the encoder
        decoder_cell: the RNN cell for the decoder
        output_proj: a tuple defining the projection of the decoder output in the form (weights, biases)
        encoder_input: the input sequence for the encoder, of the form (batch, time_step, depth)
        decoder_input: the input sequence for the decoder, of the form *batch, time_step, depth)
        is_training: whether the current mode is a training mode, defines which decoder input method
                     should be used as discussed above
        enc_sequence_length: the length of each encoder sequence, of the form (batch_size)
        dec_sequence_length: the length of each decoder sequence, of the form (batch_size)
        bucket: the size of the bucket this model is to process
        encoder_output_size: the depth of the encoder output
        attn_state_positions: optional - the positions of attention cell states, if none these states
                              are found automatically
        decoder_state_positions: optional - the positions of the base cell states in the decoder
                                 state tuple, if none these states are found automatically
        encoder_state_positions: optional - the positions of the base cell states in the encoder
                                 state tuple, if none these states are found automatically
    
    Raises:
        ValueError: if the number of encoder base states is different from the number of decoder
                    base states
    
    
    
    
    """

    batch_size = tf.shape(decoder_input)[0]
    
    # esl_assert = tf.Assert( tf.equal( tf.shape( enc_sequence_length )[0], batch_size),
    #                     [batch_size, tf.shape( enc_sequence_length )[0]])
    #                     
    # di_assert = tf.Assert( tf.equal( tf.shape( decoder_input )[0], batch_size),
    #                     [batch_size, tf.shape( decoder_input )[0]])
    #                     
    # dsl_assert = tf.Assert( tf.equal( tf.shape( dec_sequence_length )[0], batch_size),
    #                     [batch_size, tf.shape( dec_sequence_length )[0]])
    #                     
    # with tf.control_dependencies([esl_assert, di_assert, dsl_assert]):
    #     batch_size = batch_size
    
    with tf.variable_scope('encoder'):
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(encoder_cell,
                                                            encoder_input,
                                                            sequence_length = enc_sequence_length,
                                                            dtype = tf.float32)
    
    e_states_flat = nest.flatten(encoder_states)
    if attn_state_positions is None:
    #recursively find attention layers and their locations in the state tuple
        attn_state_positions = find_attn_pos(decoder_cell)
    
    if encoder_state_positions is None:
        encoder_state_positions = find_hidden_pos(encoder_cell)
    
    if decoder_state_positions is None:
        decoder_state_positions = find_hidden_pos(decoder_cell)

    if len(encoder_state_positions) != len(decoder_state_positions):
        raise ValueError("""encoder and decoder cells must have the same number of base cell states %i, %i.
                        This error means that states cannot be passed from the encoder to decoder RNNs
                        as the number of states are incompatible, you can manually set which states should
                        be passed using decoder and encoder state_positions. Cells can also be ignored
                        when finding states to pass by giving the cell a _do_not_find attribute
                        """%(len(encoder_state_positions), len(decoder_state_positions)))
    
    
    #define the loop_fn to be used with raw_rnn
    def loop_fn(time, cell_output, cell_state, loop_state):
        #cell output is unchanged
        emit_output = cell_output 
        
        #on the first step feed encoder outputs into the attention states and encoder base cell states into
        #the decoder base cell states
        if cell_state is None:
            next_state = decoder_cell.zero_state(batch_size, tf.float32)
            flat_next_state = nest.flatten(next_state)
            #feed attention inputs
            for pos in attn_state_positions:
                flat_next_state[pos] = tf.reshape(encoder_outputs, [batch_size, encoder_output_size*bucket[0]])
                flat_next_state[pos+1] = enc_sequence_length
            
            #initialise to encoder final states
            for e_pos, d_pos in zip(encoder_state_positions, decoder_state_positions):
                flat_next_state[d_pos] = e_states_flat[e_pos]
            
            next_cell_state = nest.pack_sequence_as(next_state, flat_next_state)
        else:
            next_cell_state = cell_state
        
        
        #Find the predicted symbol
        
        def n_inputs():
            #if this is not the first step
            if cell_output is not None:
                #the projection is huge so don't do it if we don't have to, 
                #this means operations must be defined inside of the function passed to tf.cond
                
                def max_symbol():
                    proj_output = tf.matmul(cell_output, tf.transpose(output_proj[0])) + output_proj[1]
                    return tf.cast(tf.argmax(emit_output, axis = 1), tf.int32)
            
                #get either the true or predicted symbol for the next input
                next_input = tf.cond(
                                    is_training,
                                    lambda: tf.reshape(decoder_input[:, time, :], [batch_size, 1]),
                                    max_symbol)
                
                #tensorflow looses track of the shape here so set it as (batch_size (may be None), 1)
                next_input.set_shape([decoder_input.get_shape().as_list()[0],1])
                
            else:
                next_input = tf.reshape(decoder_input[:, time, :], [batch_size, 1])
            
            return next_input
        
        #currently, if all the batches are labelled finished the RNN stops and the output is truncated
        #this leads to inconsistent shapes elsewhere in the model and there is no way to set a separate
        #all_finished flag. This line deals with this issue but may lead to the final state of the
        #decoder being innacurate, this state is not used at the moment but in future it may be
        #necessary to fix this
        batches_finished = tf.reduce_all(time >= tf.shape(decoder_input)[1])
        
        #if all batches are finished give zeros as next input
        next_input = tf.cond(tf.reduce_all(batches_finished),
                            lambda: tf.zeros_like(tf.reshape(decoder_input[:, 0, :], [batch_size, 1]), dtype = tf.int32),
                            n_inputs)
        
        next_loop_state = None
        
        return (batches_finished, next_input, next_cell_state, emit_output, next_loop_state)
    
    #make the rnn
    with tf.variable_scope('decoder'):
        decoder_outputs, _, _ = tf.nn.raw_rnn(decoder_cell, loop_fn) 
    
    #transpose so that time is not major (the stack operation produces as tensor with time as the major
    decoder_outputs = tf.transpose(decoder_outputs.stack(), perm = [1,0,2])
    
    return decoder_outputs
    