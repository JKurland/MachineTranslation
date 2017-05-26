import math

from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework.dtypes import int32, float32
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.contrib.layers.python.layers.layers import batch_norm
_checked_scope = core_rnn_cell_impl._checked_scope  # pylint: disable=protected-access

_linear = core_rnn_cell_impl._linear



def _state_size_with_prefix(state_size, prefix=None):
  """Helper function that enables int or TensorShape shape specification.

  This function takes a size specification, which can be an integer or a
  TensorShape, and converts it into a list of integers. One may specify any
  additional dimensions that precede the final state size specification.

  Args:
    state_size: TensorShape or int that specifies the size of a tensor.
    prefix: optional additional list of dimensions to prepend.

  Returns:
    result_state_size: list of dimensions the resulting tensor size.
  """
  result_state_size = tensor_shape.as_shape(state_size).as_list()
  if prefix is not None:
    if not isinstance(prefix, list):
      raise TypeError("prefix of _state_size_with_prefix should be a list.")
    result_state_size = prefix + result_state_size
  return result_state_size
  
             

def _state_placeholders(state_size, dtype):
  """Create placeholders for the hidden states of RNNs, can be used instead of zero_states
  Args:
    state_size: a state_size tuple or nested tuple, the placeholders will be packed in the same
                structure is this tuple
    dtype: the data type to be used
  
  """
  if nest.is_sequence(state_size):
    state_size_flat = nest.flatten(state_size)
    placeholders_flat = [
        array_ops.placeholder(dtype,
            _state_size_with_prefix(s, prefix=[None])) for s in state_size_flat
    ]
    for s, p in zip(state_size_flat, placeholders_flat):
      p.set_shape(_state_size_with_prefix(s, prefix=[None]))
    placeholders = nest.pack_sequence_as(structure=state_size,
                                  flat_sequence=placeholders_flat)
  else:
    placeholders_size = _state_size_with_prefix(state_size, prefix=[None])
    palceholders = array_ops.placeholder(dtype,placeholder_size)
    placeholders.set_shape(_state_size_with_prefix(state_size, prefix=[None]))

  return placeholders


def state_placeholders(cell, dtype):
    """creates state placeholder from RNN cell"""
    with ops.name_scope(type(cell).__name__ + "placeholder"):
      state_size = cell.state_size
      return _state_placeholders(state_size, dtype)



                
            
class AttentionWrapper(RNNCell):
    """Computes a weighted average over attn_inputs, with weights being 
        determined by a neural network within the wrapper.
        Based on https://arxiv.org/pdf/1409.0473.pdf """
    
    def __init__(self, cell, attn_input_shape, hidden_layer_size,
            activation = math_ops.tanh, input_size = None, 
            state_is_tuple = True, reuse = None):
        """ creates a cell which records its inputs and performs a weighted
        average over the stored inputs to produce attention based inputs.
        
        Args:
            cell: instance of RNNCell
            attn_input_shape: the shape of the attention input in the form (time_steps, depth)
                              should not include batch size
            hidden_layer_size: the size of the hidden layer in the neural network
            activation: the activation function for the neural network
            input_size: size of the input being expected by cell
            state_is_tuple: whether the state is stored as a tuple, must be true
            reuse: should weights be reused
                               
        Raises:
            TypeError: if cell is not an RNNCell or if attn_input is not 3d
            ValueError: if cell returns a state tuple and state_is_tuple is set
                        to false
            NotImplementedError: if state_is_tuple is False
        """
        
        if nest.is_sequence(cell.state_size) and not state_is_tuple:
            raise ValueError("Cell returns tuple of states, but the flag "
                            "state_is_tuple is not set. State size is: %s"
                            % str(cell.state_size))
    
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        
        self._input_size = input_size
        self._cell = cell
        self._state_is_tuple = state_is_tuple
        self._reuse = reuse
        self._activation = activation
        self._hidden_layer_size = hidden_layer_size
        self._attn_shape = attn_input_shape #(time_steps, depth)
        self._attn_size = self._attn_shape[0] * self._attn_shape[1]
        self._pos_in_state = len(nest.flatten(self._cell.state_size))
        self._is_attn_cell = True
        self._do_not_find = True
        
    @property
    def state_size(self):
        #add 4 new states, a flat version of the attention input, the sequence length for the attention
        #input, space for flat version of pre-computed contributions to the hidden layer, flag for 
        #whether this is the first step
        return (self._cell.state_size, self._attn_size, 1, self._attn_shape[0]*self._hidden_layer_size, 1)

    @property
    def output_size(self):
        return self._cell.output_size  
    
    def __call__(self, inputs, state, scope = None):
        with _checked_scope(self, scope or "attention_wrapper", reuse=self._reuse):
            
            if self._state_is_tuple: #unpack states
                state, attn_input_flat, attn_seq_len, input_contrib, not_first_step = state
            else:
                raise NotImplementedError
            
            #reshape flattened states
            input_contrib = array_ops.reshape(input_contrib,
                        [-1, self._attn_shape[0], self._hidden_layer_size])
                        
            attn_input = array_ops.reshape(attn_input_flat,
                        [-1, self._attn_shape[0], self._attn_shape[1]])
            
            with vs.variable_scope('filters'):
                filter2 = vs.get_variable('filter2', shape = [1, self._hidden_layer_size, 1])
                filter1 = vs.get_variable('filter1', shape = [1, self._attn_shape[1], self._hidden_layer_size])

            """ on the first step calculate contributions from the attention input, this will be 
            the same on every subsequent step so there is no need to recompute every step
            
            """
            def compute_input_contrib():
                return nn_ops.conv1d(attn_input, filter1, 1, "SAME")
            
            input_contrib = control_flow_ops.cond(
                                    math_ops.reduce_all(math_ops.equal(not_first_step, 1)),
                                    lambda: input_contrib,
                                    compute_input_contrib)
            
            """ get the hidden state (c_state for LSTMs) of the original cell
             or the lowest cell in a MultiRNN stack, deals with possible previous
            wrappings, if using other states is necessary find_hidden_pos from rnns.py could
            be used"""
            
            s = state
            while nest.is_sequence(s):
                s = s[0]
                
            prev_state = s
            with vs.variable_scope('weights1'):
                #compute the contribution from the hidden state, this must be done every step
                state_contrib = _linear(s,self._hidden_layer_size, True)
            
            hidden = self._activation(array_ops.expand_dims(state_contrib,1) + input_contrib)
            
            #conv1d is used as the same weights are used for each time step, so the filter in convulved
            #through time
            energy = nn_ops.conv1d(hidden, filter2, 1, "SAME")
            
            weights = nn_ops.softmax(energy)
            
            # remove weights that are out of sequence
            mask = array_ops.sequence_mask(attn_seq_len, maxlen = self._attn_shape[0])
            mask = array_ops.expand_dims(mask,2)
            weights = array_ops.where(mask, weights, array_ops.zeros_like(weights))
                
            # renormalise the weights so that the weighted average isn't biased towards 0
            weights = weights / math_ops.reduce_sum(weights, axis = 1, keep_dims = True)
            
            #calculate the weighted average
            inputs_from_attn = array_ops.reshape(math_ops.reduce_sum( weights*attn_input, axis = 1), [-1, self._attn_shape[1]])
            
            input_size = self._input_size
            if input_size is None:
                input_size = inputs.get_shape().as_list()[1]
            
            with vs.variable_scope('weights2'):
                #project the weighted average so it is the correct size for the cell being wrapped
                inputs = _linear([inputs, inputs_from_attn], input_size, False) 
            
            outputs, state = self._cell(inputs, state)
            #flatten input_contrib to be packed into state
            input_contrib = array_ops.reshape(input_contrib,
                            [-1, self._attn_shape[0]*self._hidden_layer_size])
            
            state = (state,
                    attn_input_flat,
                    attn_seq_len, 
                    input_contrib, 
                    array_ops.ones_like(not_first_step))
                
            return outputs, state
            
class DropoutWrapper(RNNCell):
    """simple dropout layer for cells, only applies dropout to the cell output, not the cell state"""
    
    def __init__(self, cell, keep_prob):
        """makes the cell
        Args:
            cell: the parent cell to be wrapped
            keep_prob: the keep probability
            """
        
        self._cell = cell
        self._keep_prob = keep_prob
    
    @property
    def state_size(self):
        return self._cell.state_size
    
    @property
    def output_size(self):
        return self._cell.output_size
    
    def __call__(self, input, state, scope = None):
        with vs.variable_scope(scope or 'dropout'):
            output, state = self._cell(input, state)
            
            output = nn_ops.dropout(output, self._keep_prob)
            
            return (output, state)
           
    
class AttentionCell(RNNCell):
    """ a cell which performs the attention layer in
        http://aclweb.org/anthology/D15-1166, there are threee different
        attention weight computation methods "dot", "general" (quadratic) and 
        "linear concatenation". The dot method requires that the source and target
        RNNs have the same layer size. This class assumes that time_major is false,
        that is that the form of tensors with a time component is 
        (batch, time, depth)
    """
    
    def __init__(self, method, input_size, attented_shape, input_bypass = True,
                project_output = False, projection_size = None, nn_hidden_size = None,
                nn_activation = math_ops.tanh, reuse=None):
        """ creates the attention cell
        
        Args:
            method: one of 'dot', 'quad' or 'lin'
                'dot': e_s,t = dot(attn_s,h_t)
                'quad': e_s,t = transpose(attn_s) * W * h_t (transpose(attn_s)*W can be precomputed)
                'lin': e_s,t = W*[attn_s,h_t]
                'nn': e_s,t = nn([attn_s,h_t])
                where attn_s is the attention input from encoder time step s, h_t is the input to this
                cell and e_s,t is the energy for encoder time step s and current time step t
            input_size: the size of the input to this cell
            attented_shape: the shape of the vectors being attented over in the
                            form (time_steps, depth), the context vector will
                            be of size (depth)
            input_bypass: whether or not to bring the input of the layer through
                  the cell and concatenate it with the calculated context vector
                    defaults to True
            project_output: whether or not to project the output of the cell
                            defaults to False. This should be used if the input
                            and output of the cell must be the same size and
                            input bypass is used
            projection_size: the size of the projected output
            nn_hidden_size: if method is 'nn' this specifies the size of the hidden layer of the nn
            nn_activation: the activation function for the neural network
        Raises:
            TypeError: if method is not a string
            ValueError: if project_output is True and projection_size is None
                        or if method is nn and nn_hidden_size is None
                        or if method is not one of the above options
                        or if method is dot and input_size and attented_shape
                            are incompatible
        
        """
        if type(method) is not str:
            raise TypeError('method argument must be a string')
        if method not in ['dot', 'quad', 'lin', 'nn']:
            raise ValueError('method given (%s) not in allowed values %s'%(method, 'dot, quad, lin'))
        if project_output and (projection_size is None):
            raise ValueError('project_output set to true but no projection size given')
        if method is 'dot' and input_size != attented_shape[1]:
            raise ValueError("""input_size and attented_shape depth must match
            when method is 'dot', got %i, %i"""%(input_size, attented_shape[1]))
        if method is 'nn' and nn_hidden_size is None:
            raise ValueError('is method "nn" is being used a hidden layer size must be specified')
        
        self._method = method
        self._attented_shape = attented_shape
        self._attented_size = attented_shape[0]*attented_shape[1]
        self._input_bypass = input_bypass
        self._project_output = project_output
        self._projection_size = projection_size
        self._input_size = input_size
        self._nn_hidden_size = nn_hidden_size
        self._nn_activation = nn_activation
        self._reuse = reuse
        #set the is_attn_cell flag so this cell can be found by the RNN function,
        #the cell is then given the data to be attented over in the first state
        #slot and the sequence lengths in the second.
        self._is_attn_cell = True
        #set this flag so this cell isn't found when passing states from the encoder to decoder stage
        self._do_not_find = True
        
        
    @property
    def state_size(self):
        if self._method is 'dot':            
            return (self._attented_size, 1)
        if self._method is 'lin':
            #space for saving reusable computation
            return (self._attented_size, 1, self._attented_shape[0], 1)
        if self._method is 'quad':
            return (self._attented_size, 1, self._attented_shape[0]*self._input_size, 1)
        if self._method is 'nn':
            return (self._attented_size, 1, self._attented_shape[0]*self._nn_hidden_size, 1)
        
    @property
    def output_size(self):
        if self._project_output:
            return self._projection_size
        if self._input_bypass:
            return self._input_size + self._attented_shape[1]
        return self._attented_shape[1]
    
    def __call__(self, input, state, scope = None):
        with vs.variable_scope(scope or 'attention_cell', reuse = self._reuse):
            
            #unpack the state tuple
            attn_input_flat = state[0]
            attn_seq_len = state[1]
            
            attn_input = array_ops.reshape(attn_input_flat, [-1] + self._attented_shape)
            
            if self._method in ['lin', 'quad', 'nn']:
                input_contrib_flat = state[2]
                not_first_step = state[3] # is zero on the first step
        
            #linear method
            if self._method is 'lin':
                #input_contrib = array_ops.expand_dims(input_contrib_flat, 2)
                input_contrib = input_contrib_flat
                
                with vs.variable_scope('filters'):
                    filter = vs.get_variable('lin_filter', shape = [1, self._attented_shape[1], 1])
                    
                #if this is the first step, compute attention contributions
                input_contrib = control_flow_ops.cond(
                                        math_ops.reduce_all(math_ops.equal(not_first_step, 0)),
                                        lambda: array_ops.reshape(nn_ops.conv1d(attn_input, filter, 1, "SAME"),[-1,self._attented_shape[0]]),
                                        lambda: input_contrib, name='lin_cond')
                                        
                energies = array_ops.expand_dims(input_contrib, 2) + array_ops.expand_dims(_linear(input, 1, False),1)
                
                state = (attn_input_flat, attn_seq_len, input_contrib, array_ops.ones_like(not_first_step))
            
            #quadratic method
            if self._method is 'quad':
                input_contrib = array_ops.reshape(input_contrib_flat,[-1, self._attented_shape[0], self._input_size])
                
                with vs.variable_scope('filters'):
                    filter = vs.get_variable('quad_filter', shape = [1, self._attented_shape[1], self._input_size])
                    
                input_contrib = control_flow_ops.cond(
                                        math_ops.reduce_all(math_ops.equal(not_first_step, 0)),
                                        lambda: nn_ops.conv1d(attn_input, filter, 1, "SAME"),
                                        lambda: input_contrib)
                
                energies = math_ops.reduce_sum(input_contrib*array_ops.expand_dims(input, 1),
                                               2, keep_dims = True)
                                               
                input_contrib_flat = array_ops.reshape(input_contrib, [-1, self._attented_shape[0]*self._input_size])
                state = (attn_input_flat, attn_seq_len, input_contrib_flat, array_ops.ones_like(not_first_step))
                
            #dot method
            if self._method is 'dot':
                energies = math_ops.reduce_sum(attn_input*array_ops.expand_dims(input, 1),
                                               2, keep_dims = True)
                state = (attn_input_flat, attn_seq_len)
            
            #neural method
            if self._method is 'nn':
                input_contrib = array_ops.reshape(input_contrib_flat,[-1, self._attented_shape[0], self._nn_hidden_size])
                
                with vs.variable_scope('filters'):
                    filter1 = vs.get_variable('nn_filter1', shape = [1, self._attented_shape[1], self._nn_hidden_size])
                    filter2 = vs.get_variable('nn_filter2', shape = [1, self._nn_hidden_size, 1])
                    
                input_contrib = control_flow_ops.cond(
                                        math_ops.reduce_all(math_ops.equal(not_first_step, 0)),
                                        lambda: nn_ops.conv1d(attn_input, filter1, 1, "SAME"),
                                        lambda: input_contrib)
                
                hidden = input_contrib + array_ops.expand_dims(_linear(input, 1, True),1)
                         
                hidden = self._nn_activation(hidden)
                
                energies = nn_ops.conv1d(hidden, filter2, 1, "SAME")
                
                input_contrib_flat = array_ops.reshape(input_contrib, [-1, self._attented_shape[0] * self._nn_hidden_size])
                state = (attn_input_flat, attn_seq_len, input_contrib_flat, array_ops.ones_like(not_first_step))
                
            
            energies.set_shape([None, self._attented_shape[0], 1])
            weights = nn_ops.softmax(energies, dim = 1)
        
            # remove weights that are out of sequence
            mask = array_ops.sequence_mask(attn_seq_len, maxlen = self._attented_shape[0])
            mask = array_ops.expand_dims(mask,2)
            weights = array_ops.where(mask, weights, array_ops.zeros_like(weights))
                
            # renormalise the weights so that the weighted average isn't biased towards 0
            weights = weights / (math_ops.reduce_sum(weights, axis = 1, keep_dims = True) + 0.0000001)
            
            context = array_ops.reshape(math_ops.reduce_sum( weights*attn_input, axis = 1), [-1, self._attented_shape[1]])
            
            if self._input_bypass:
                output = array_ops.concat([context, input], 1)
            else:
                output = context
                
            if self._project_output:
                with vs.variable_scope('output_projection'):
                    output = _linear([output], self._projection_size, True)
            
            return (output, state)

#this is almost a copy of the tf embedding wrapper with a minor fix, I've made a pull request on github but it hasn't gone through yet
class EmbeddingWrapper(RNNCell):
  """Operator adding input embedding to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the embedding on this batch-concatenated sequence, then split it and
  feed into your RNN.
  """

  def __init__(self, cell, embedding_classes, embedding_size, initializer=None,
               reuse=None):
    """Create a cell with an added input embedding.

    Args:
      cell: an RNNCell, an embedding will be put before its inputs.
      embedding_classes: integer, how many symbols will be embedded.
      embedding_size: integer, the size of the vectors we embed into.
      initializer: an initializer to use when creating the embedding;
        if None, the initializer from variable scope or a default one is used.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if embedding_classes is not positive.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    if embedding_classes <= 0 or embedding_size <= 0:
      raise ValueError("Both embedding_classes and embedding_size must be > 0: "
                       "%d, %d." % (embedding_classes, embedding_size))
    self._cell = cell
    self._embedding_classes = embedding_classes
    self._embedding_size = embedding_size
    self._initializer = initializer
    self._reuse = reuse

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def __call__(self, inputs, state, scope=None):
    """Run the cell on embedded inputs."""
    with _checked_scope(self, scope or "embedding_wrapper", reuse=self._reuse):
      with ops.device("/cpu:0"):
        if self._initializer:
          initializer = self._initializer
        elif vs.get_variable_scope().initializer:
          initializer = vs.get_variable_scope().initializer
        else:
          # Default initializer for embeddings should have variance=1.
          sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
          initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)

#minor fix to allow multiple wrappers
        s = state
        while type(s) is tuple:
          s = s[0]
        data_type = s.dtype

        embedding = vs.get_variable(
            "embedding", [self._embedding_classes, self._embedding_size],
            initializer=initializer,
            dtype=data_type)
        embedded = embedding_ops.embedding_lookup(
            embedding, array_ops.reshape(inputs, [-1]))
    return self._cell(embedded, state)
    
class BatchNormWrapper(RNNCell):
    """a wrapper which batch normalises the output of an RNN cell"""
    
    def __init__(self, cell, is_training):
        """makes the wrapper cell
        Args:
            cell: RNNCell which is to be wrapped
            is_training: a bool type tensor which is true during training, and false during inference
                        During training the wrapper normalises based on the current batch, otherwise
                        a moving average is used.
        Raises:
            TypeError: if is_training is not boolean
            
        returns: the new wrapped cell
        """
        
        self._cell = cell
        self._is_training = is_training
    
    @property
    def state_size(self):
        return self._cell.state_size
    
    @property
    def output_size(self):
        return self._cell.output_size
    
    def __call__(self, inputs, states, scope = None):
        with vs.variable_scope(scope or 'batch_normalisation'):
            
            output, states = self._cell(inputs, states)
            output = batch_norm(output,
                                is_training = self._is_training,
                                reuse = True, scope = 'batch_norm', 
                                zero_bias_moving_mean = True)
            
            return output, states
            
            
            
            
            
            
            
            
            
            
            
            