import sys
from math import ceil

from task_1_model import *
from get_data import *




PAD = 0
GO = 1
EOS = 2
UNK = 3




help_string = """set model and training parameters and file paths from this command line, possible commands are:\n
                set: sets a parameter to a value, example: set learning_rate 1e-3\n
                list: list either the current paramters or files, examples: list params, list files\n
                clear_buckets: remove all the current buckets\n
                add_bucket: add a new bucket, example: add_bucket (6,7). n.b. the bucket parameter can be set using set
                    by using paramter name bucket and giving a list of tuples\n
                set_file: sets the path to a file, example: set_file source_ids giga-fren.release2.fixed.en.ids40000\n
                show: shows the value of a parameter, example: show layer_size\n
                fit: rus the estimator.fit method, this trains the model, the number of steps is set using the steps parameter\n
                eval: evaluates the model\n
                decode: translate a phrase with the current model, example: decode these are some english words\n
                look_up: looks up an index in the vocabulary, example loop_up 265\n
Commands can be run automatically at startup by creating a file called "com" in the data or logs directory. Each line of this\n
file should be an instance of one of the commands above, each line of "com" will be run in order. If a com file exists in both data\n
and logs, the data com file is run first, and then the logs file. More commands can be entered after the "com" file is finished.

                """
    
    
def model_fn(features, labels, mode, params, config):
    """model function to be passed to tf.contrib.learn.Estimator
    
    Args:
        features: dictionary of feature tensors 
        labels: dictionary of label tensors
        mode: mode that the estimator is being run in (train, evaluate or infer)
        params: dictionary of model hyper-parameters, these are:
            
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
            
        config: configuration object, will receive what is passed to Estimator in the config argument
    
    returns: ModelFnOps object
    
    
    """
    buckets = params['buckets']
    
    #set is_training (used to determine whether the decoder input will be the true symbol or the
    #previously estimated symbol, also for batch_norm if used) and keep_prob based on the operating
    #mode
    if mode is tf.contrib.learn.ModeKeys.TRAIN:
        params['is_training'] = tf.constant(True, dtype = tf.bool)
        params['keep_prob'] = params['keep_prob']
    else:
        params['is_training'] = tf.constant(False, dtype = tf.bool)
        params['keep_prob'] = 1.0
    
    
    # make the actual model
    with tf.variable_scope('model'):
        
        token_prediction, loss, num_non_zero = create_model(features,
                                                            labels,
                                                            params,
                                                            mode)
    
    #if the network is not being run in inference mode, set up errors and training ops
    if mode != tf.contrib.learn.ModeKeys.INFER:
        
        #total loss is the weighted average of the bucket losses, based on the number of non zero (pad)
        #symbols in each bucket
        total_loss = sum(loss)/tf.cast(sum(num_non_zero),tf.float32)
        loss = [l/(tf.cast(n, tf.float32) + 0.0001) for l,n in zip(loss, num_non_zero)]
        
        #get the global step and create increment operation
        step = tf.train.get_global_step()
        inc_step = tf.assign_add(step,1)
        
        #make training ops
        with tf.variable_scope('training'): 
            
            
            #get regularisation losses
            lam = params['l2_reg_strength']
            
            #check if a variable is a weight using its shape
            def var_is_weight(var):
                shape = var.get_shape().as_list()
                #short circuit to avoid out of range error
                return (len(shape) != 0) and ((shape[0]!=1 and len(shape)!=1) or len(shape)>2)
            
            #ignore biases
            reg_loss = sum([tf.nn.l2_loss(var) for var in tf.trainable_variables() if var_is_weight(var)])
            
            #include biases
            #reg_loss = sum([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            
            log_alpha = tf.Variable(tf.log(params['alpha']), name = 'alpha')
            
            #compute gradient and clip by norm
            opt = tf.train.AdamOptimizer(tf.exp(log_alpha))
            grads_vars = opt.compute_gradients(total_loss + lam*reg_loss)
            clipped_grads, _ = tf.clip_by_global_norm([a for a,b in grads_vars], params['max_grad_norm'])
            clipped_grads_vars = [(grad, var) for grad,var in zip(clipped_grads, [b for a,b in grads_vars])]
            
            #two complex options for learning rate control:
            #   1. keep a moving record of previous losses, if the current loss if greater than all
            #      the recent losses reduce the learning rate
            #   2. find the error gradient with respect to the learning rate using chain rule;
            #                   dJ/d(log(alpha)) = sum(dJ/d(parameter) * d(parameter)/d(alpha) * (d(exp(log_alpha))/d(log_alpha) = alpha)))
            #                   d(parameter)/d(alpha) = (parameter_t - parameter_{t-1})/alpha
            #       this indirect method of calculating d(param)/d(a) is due to the optmiser being
            #       not very transparent when it comes to the updates it applies. 
            #(possibly search for ADAM parameters using list comprehension then compute directly)
            #       this ignores second order information, then use ADAM to optimise the learning rate
            #       it is similar to the method used here  https://arxiv.org/pdf/1606.04474v1.pdf
            
            #previous losses method
            # p_losses = tf.Variable(tf.zeros([50], tf.float32))
            # new_alpha = tf.cond(tf.logical_and(step>10, tf.reduce_max(p_losses) < total_loss),lambda: alpha*params['alpha_decay'], lambda: alpha)
            # update_alpha = tf.assign(alpha, new_alpha)
            # with tf.control_dependencies([update_alpha]): #ensure that alpha updates before p_losses
            #     update_p_losses = tf.assign(p_losses, tf.concat([tf.slice(p_losses, [1], [-1]), tf.expand_dims(total_loss,0)], 0))
            

            #optimising alpha method (possibly try adding an l2 regulariser to alpha so that easy
            #parameters aren't exploited at the expense of more curvy, difficult parameters)
            p_vars = {}
            update_p_vars = []
            for _,var in grads_vars:
                p_vars[var] = tf.Variable(tf.zeros_like(var, dtype = var.dtype), trainable = False)
                update_p_vars.append(tf.assign(p_vars[var], var))
            
            alpha_alpha = tf.Variable(0.0) #the learning rate for the learning rate
            alpha_opt = tf.train.AdamOptimizer(alpha_alpha)
            var_delta_grad = [(var - p_vars[var], grad) for grad,var in grads_vars]
            
            #the gradient for alpha is very large at the start of training, ignore these gradient
            #by keeping the learning rate's learning rate at 0 until step 100
            def set_alpha_alpha():
                with tf.control_dependencies([tf.assign(alpha_alpha, params['alpha_alpha'])]):
                    return tf.zeros([0])
                    
            def no_set():
                return tf.zeros([0])
                
            alpha_alpha_set = tf.cond(tf.greater(step, 10), set_alpha_alpha, no_set)
            
            #set alpha_alpha, then update alpha
            with tf.control_dependencies([alpha_alpha_set]):
                alpha_train_op = alpha_opt.apply_gradients([(sum([tf.reduce_sum(delta*grad) for delta, grad in var_delta_grad if not grad is None]) + 0.5*params['l2_reg_strength_alpha']*tf.exp(log_alpha), log_alpha)])
            
            #increment global step, do alpha_training and update previous variable values before 
            #applying gradients
            with tf.control_dependencies([inc_step, alpha_train_op] + update_p_vars):
                train_op = opt.apply_gradients(clipped_grads_vars)
        
        #make summaries and the loss dictionary
        loss_dict = {}
        loss_summaries = []
        
        for i,l,bucket in zip(range(len(buckets)),loss,buckets):
            name = 'bucket_{}_loss'.format(i)
            loss_dict[name] = l
            loss_summaries.append(tf.summary.scalar(name, l))
            
        total_loss_summary = tf.summary.scalar('total_loss', total_loss)
        loss_dict['total_loss'] = total_loss
        
        alpha_summary = tf.summary.scalar('learning_rate', tf.exp(log_alpha))
    else:
        #if running in inference mode
        total_loss = None
        train_op = None
        loss_dict = None
    
    all_summaries = tf.summary.merge_all()
    
    pred_dict = {}
    for i,bucket in enumerate(buckets):
        pred_dict[bucket] = token_prediction[i]
    
    #state serialization for TextLineReader is currently not implemented, this means the file reader
    #state cannot be restored, this means the reader will restart from the start of the file
    #whenever the fit operation is run. This means it is important to run full epochs at a time so 
    #that the model does not overfit to samples near the start of the file.
    #local_init_op is run after a session restore has been attempted, this means that variables are
    #either restored or initiliased. If the initialiser has been run then this operation should not
    #be run
    # restore_readers = tf.get_collection(tf.GraphKeys.INIT_OP)
    # with tf.control_dependencies(restore_readers):
    #     local_init_op = tf.no_op
        
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    scaffold = tf.train.Scaffold(saver = saver,
                                 init_op = init,
                                 summary_op = all_summaries)
                                
    
    # set up the queue runner start hook, this runs after the session had been created
    try:
        qr = params['qr'][0]
    except:
        qr = None
    
    #make training_hook
    class QrStartHook(tf.train.SessionRunHook):
        def __init__ (self, qr):
            self._qr = qr
        
        def after_create_session(self, session, coord):
            """starts the queue runner when the session is created"""
            threads = self._qr.create_threads(session, coord = coord, start = True)
            tf.train.start_queue_runners(sess = session, coord = coord)
            
    if qr:
        training_hooks = [QrStartHook(qr)]
    else:
        training_hooks = None
    
    return tf.contrib.learn.ModelFnOps(mode = mode,
                                      predictions = pred_dict,
                                      loss = total_loss,
                                      train_op = train_op,
                                      scaffold = scaffold,
                                      eval_metric_ops = loss_dict,
                                      training_hooks = training_hooks
                                      )
                                      
def make_input_fn(batch_size, source_path, target_path, queue_capacity, buckets, qr_list, validate_buckets = True):
    """makes the input_fn to be used with the estimator class. The input_fn returns two dictionaries,
    features and labels. Features contains the inputs to the model, labels contains the target outputs.
    
    This input function created a shuffle queue, that is, a queue which dequeues random elements as
    opposed to first in first out. Each element of the queue is a list with length 2*num_buckets,
    one encoder and one decoder element per bucket. 
    
    Two queue runners are necessary to read data from the files, combine the data, then add the data
    to the queue.
    
    Args:
        batch_size: the number of samples contained in each mini-batch, this is the total number of
                    samples across all buckets
        source_path: the path of the source IDs file, the path is relative to data/ . So if the full
                     file path is data/file/path, then this function will take only file/path. This
                     is to keep the data and logs directories separate and make operation simpler when
                     using the docker image. An absolute path can be specified using /path/to/file
        target_path: the path of the target IDs file, follows the same pattern as source_path
        queue_capacity: the maximum number of items in the shuffle queue at once, the larger this number
                        the more well mixed the samples will be. Must be larger than batch_size, should
                        be larger than 2*batch_size
        buckets: a list of bucket shapes, the buckets must be ordered by size or other preference,
                with the smallest or most prefered at the start. If bucket preference is something other
                than size validate_buckets should be set to false
        validate_buckets (default True): Check if the list of buckets is in size order, raise ValueError
                                         if they are not.
                                         
        Raises:
            ValueError: if validate_buckets is True and buckets are not in size order
                        or if batch_size is greater than queue_capacity
            FileNotFoundError: if either of the id files or not found
    
    """
    def input_fn():
        INF = tf.convert_to_tensor(2147483647, dtype = tf.int32) #higher than any other int32
        with tf.variable_scope('Input_Fn'):
            full_source_path = os.path.join('data',source_path)
            full_target_path = os.path.join('data',target_path)
            
            #check if buckets are sorted by size
            if validate_buckets and not all([sum(a) <= sum(b) for a,b in zip(buckets[:-1],buckets[1:])]):
                raise ValueError ("""Bucket list is not sorted by size, either sort the buckets or set 
                                    validate_buckets to False, this could lead to poor bucket choices""")
                
            if not os.path.isfile(full_source_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), full_source_path)
                
            if not os.path.isfile(full_target_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), full_target_path)
                
            if batch_size > queue_capacity:
                raise ValueError("Batch size must be less than the queue capcity (%i, %i)"%(batch_size, queue_capacity))
            
            if batch_size*2 > queue_capacity:
                print("""Warning: batch_size should not be greater than half the queue_capacity, this could
                        allow the queue to be emptied completely, or may result in poor mixing of samples""")
                
        
            
            
            #connect tensors (source_value and target_value) to the source and target files
            source_filename_queue = tf.train.string_input_producer([full_source_path])
            target_filename_queue = tf.train.string_input_producer([full_target_path])
            
            source_reader = tf.TextLineReader()
            target_reader = tf.TextLineReader()
            
            #TextLineReader state serialization is currently not implemented, the following commented
            #lines would use state serialization to restore the file reader state, meaning the file
            #reader would start at the line it left off on after a restart
            
            # source_reader_state = tf.Variable(source_reader.serialize_state(), dtype=tf.string, name = 'source_reader_state', trainable = False)
            # target_reader_state = tf.Variable(source_reader.serialize_state(), dtype=tf.string, name = 'target_reader_state', trainable = False)
            # 
            # restore_source_reader = source_reader.restore_state(source_reader_state)
            # restore_target_reader = target_reader.restore_state(target_reader_state)
            # 
            # tf.add_to_collection(tf.GraphKeys.INIT_OP, restore_source_reader)
            # tf.add_to_collection(tf.GraphKeys.INIT_OP, restore_target_reader)
            
            # with tf.control_dependencies([
            #                         tf.assign(source_reader_state, source_reader.serialize_state()),
            #                         tf.assign(target_reader_state, target_reader.serialize_state())]):
                # _, source_value = source_reader.read(source_filename_queue)
                # _, target_value = target_reader.read(target_filename_queue)
                
            #read the file line into a tensor
            _, source_value = source_reader.read(source_filename_queue)
            _, target_value = target_reader.read(target_filename_queue)
            
            #convert from the space deliminated string to an integer array, then add the EOS tokens at the end of the sequence
            source_value = tf.string_split([source_value])
            source_value = tf.sparse_tensor_to_dense(source_value, default_value = b'0')
            source_value = tf.string_to_number(source_value, out_type = tf.int32)
            source_input_sample = tf.concat( [tf.reshape(source_value, [-1]), tf.reshape(tf.convert_to_tensor(EOS), [1])], 0)
            
            #same but for target line
            target_value = tf.string_split([target_value])
            target_value = tf.sparse_tensor_to_dense(target_value, default_value = b'0')
            target_value = tf.string_to_number(target_value, out_type = tf.int32)
            target_input_sample = tf.concat( [tf.reshape(target_value, [-1]), tf.reshape(tf.convert_to_tensor(EOS), [1])], 0)
            
            num_buckets = len(buckets)
            
            #create the queue, each item in the queue is a list of tensors, the signiture of this list is:
            # [bucket0_source, ... , bucketN_source, ... , bucket0_target, ... , bucketN_target]
            #as each element of the queue is only a single sample all but one of the buckets will be unused.
            #the unused buckets contain a tensor of shape [0]
            queue = tf.RandomShuffleQueue(queue_capacity, min(queue_capacity,2*batch_size), [tf.int32]*2*num_buckets)
            tf.summary.scalar('queue_size', queue.size())
            #go through every bucket, add a 1d 0 length tensor to buckets not being used, and add the padded
            #sample to the bucket being used. Add the list of tensors to a shuffle queue, when tensors are 
            #dequeued from the shuffle queue they will be concatenated, the 0 length tensors will not effect
            #anything, leaving as list of concatenated tensors ready to be reshaped.
            
            #buckets must be ordered from smallest to largest size, then each bucket is checked in order, once
            #a bucket which is large enough to fit the sample is found the source_length and target_length
            #variables are set to inf, meaning that all larger buckets will be rejected.
            
            #initialise the lengths to the actual values
            source_length = tf.shape(source_input_sample, out_type = tf.int32)[0]
            target_length = tf.shape(target_input_sample, out_type = tf.int32)[0]
            
            tensor_list = [None]*num_buckets*2
            for i, bucket in enumerate(buckets):
                
                def true_fn():
                    
                    source_padding = tf.zeros([bucket[0] - source_length], dtype = tf.int32)
                    target_padding = tf.zeros([bucket[1] - target_length], dtype = tf.int32)
                    
                    source = tf.concat([source_input_sample, source_padding], 0)
                    target = tf.concat([target_input_sample, target_padding], 0)
                    
                    return source, target, INF, INF
                    
                
                def false_fn():
                    
                    source = tf.zeros([0], dtype = tf.int32)
                    target = tf.zeros([0], dtype = tf.int32)
                    
                    return source, target, source_length, target_length
                
                source, target, source_length, target_length = tf.cond(
                                            tf.logical_and(source_length < bucket[0], target_length < bucket[1]),
                                            true_fn,
                                            false_fn)
                
                
                tensor_list[i] = source
                tensor_list[i+num_buckets] = target
            
            #make enqueue operation so that the list is only added if it has a used bucket. This allows
            #better control over the batch size 
            def add_list():
                enq = queue.enqueue(tensor_list)
                with tf.control_dependencies([enq]):
                    return tf.zeros([0])
                
            def no_add_list():
                return tf.zeros([0])
            
            enqueue_op = tf.cond(tf.equal(source_length, INF), add_list, no_add_list)
            
            
            #queue.deqeuue_many currently does not support inconsisten tensor shapes so instead make a list
            #of dequeue operations
            dq = [queue.dequeue() for _ in range(batch_size)]
            #concatenate the dequeued samples to make the encoder input and the decoder labels
            #this is done instead of stacking due to inconsistent lengths (either the correct length
            # or empty)
            x = []
            for i in range(2*num_buckets):
                x.append(tf.concat([ sample[i] for sample in dq ], 0))
                
            #create the features and labels dictionaries
            features = {}
            labels = {}
            for i, bucket in enumerate(buckets):
            
                
                features[i] = tf.reshape(x[i], [-1, bucket[0], 1])
                labels[i] = tf.reshape(x[i+num_buckets], [-1, bucket[1], 1])
                
                bucket_batch = tf.shape(features[i])[0]
                
                #need to go from the decoder labels to the decoder inputs, this means replacing
                #the EOS token with a PAD token, shifting the entire sample 1 to the right, and adding
                #a GO token to the start
                
                #remove the EOS token from the decoder inputs and replace it with the PAD token
                No_EOS = tf.where(tf.equal(labels[i], EOS),
                                        PAD*tf.ones_like(labels[i], dtype = tf.int32),
                                        labels[i])
                #add a GO token to the start of decoder inputs
                No_EOS = tf.slice(No_EOS, [0,0,0], [-1, bucket[1]-1, -1])
                
                features[i+num_buckets] = tf.concat([ tf.tile([[[GO]]], [bucket_batch, 1, 1]), No_EOS], 1)
                
                #flatten labels to be used with softmax
                labels[i] = tf.reshape(labels[i], [-1,1])
                
            #make the queue runner, and start it
            qr = tf.train.QueueRunner(queue, [enqueue_op])
            
            #a hook is necessary to start and end the runners, however, hooks can only be created in
            #the model_fn, this transfers the qr object to the model_fn, through param_dict
            #qr_list is also passed to model_fn and as the list is not copied any changes here
            #will remain in model_fn
            qr_list.append(qr) 
    
            
            
            return features, labels
        
    return input_fn
        

def print_dict(dict):
    """Prints a dictionary as a table
    Args:
        dict: Dictionary to be printed
    """
    #find the longest element in keys and values
    align = [0,0]
    for k, v in dict.items():
        if len(str(k)) > align[0]:
            align[0] = len(str(k))
        if len(str(v)) > align[1]:
            align[1] = len(str(v))
        
    for k, v in dict.items():
        print( ('{:<%i} {:<%i}'%(align[0]+4, align[1]+4)).format(str(k),str(v)))


def sort_buckets(buckets):
    """ sorts buckets by total size, buckets are used based on preference. The most prefered bucket
        should be at the start of the list, this function prioritises the smallest bucket to minimise
        memory usage
        Args:
            buckets: buckets to be sorted
        returns:
            sorted list of buckets
    """
    total_bucket_sizes = [s + t for s,t in buckets]
    return [bucket for _, bucket in sorted(zip(total_bucket_sizes, buckets), key = lambda pair: pair[0])]

if __name__ == '__main__':
    #set some default parameters
    buckets = [(7,10),(13,20)]
    param_dict = {'layer_size': 300,
                'num_layers': 3,    
                'input_symbols': 40000,
                'output_symbols': 40000,
                'embedding_size': 300,
                'attn_layer_size': 30,
                'alpha': 1e-3,
                'alpha_decay': 1.0,
                'max_grad_norm': 2.0,
                'softmax_sample_size':500,
                'keep_prob':0.5,
                'buckets': buckets,
                'steps' :2000,
                'task': 3,
                'max_samples_to_load': 3000,
                'batch_size': 64,
                'attn_method': 'nn',
                'l2_reg_strength': 4e-5,
                'alpha_alpha': 0.1,
                'l2_reg_strength_alpha': 1e-3}
    
    #set some default files
    file_dict = {'source_ids': 'giga-fren.release2.fixed.en.ids%i'%param_dict['input_symbols'],
                 'target_ids': 'giga-fren.release2.fixed.fr.ids%i'%param_dict['output_symbols'],
                 'test_source_ids': 'giga-fren.release2.fixed.en.ids%i'%param_dict['input_symbols'],
                 'test_target_ids': 'giga-fren.release2.fixed.fr.ids%i'%param_dict['output_symbols'],
                 'source_vocab': 'vocab%i.from'%param_dict['input_symbols'],
                 'target_vocab': 'vocab%i.to'%param_dict['output_symbols'],
                 'corpus_file': 'training-giga-fren.tar',
                 'unzipped_source': 'giga-fren.release2.fixed.en',
                 'unzipped_target': 'giga-fren.release2.fixed.fr'}
    
    file_dict_changes = {} #keep track of whether file names have been changed by the user
    for k in file_dict.keys():
        file_dict_changes[k] = False
    
    
    run_config = tf.contrib.learn.RunConfig(save_summary_steps = 10,
                                            model_dir = 'logs',
                                            save_checkpoints_secs = None,
                                            save_checkpoints_steps = 500)
                                        
    estimator = tf.contrib.learn.Estimator(model_fn, params = param_dict, config = run_config)
    
    #welcome message, should be made more welcoming
    print('use set param value to set parameters, list to list current parameters and values and run to start the trainer, type help for help')
    
    #load the com file if it exists, com file in data is performed first, them com in logs. This
    #allows a data specific set of commands to be run, then model specific commands to be run
    try:
        with open('data/com') as f:
            coms = f.read().splitlines()
    except:
        coms = []
    
    try:
        with open('logs/com') as f:
            coms += f.read().splitlines()
    except:
        coms = coms
    commands = 0
    
    
    while True:
        #if there are commands in the com file that have not yet been run, run the next command
        if len(coms) > commands:
            com = coms[commands]
        else: #otherwise get a new command from the user
            com = input(">  ")
        commands += 1
        
        #parse the command, simply split using spaces
        sep = com.split()
        
        #do the command
        
        if sep[0] == 'list': #list current parameters or files
            if len(sep) < 2:
                print('list takes one argument, "files" or "params"')
            else:
                if sep[1] == 'params':
                    print_dict(param_dict)
                    
                elif sep[1] == 'files':
                    align_0 = max([len(f) for f, _ in file_dict.items()])
                    align_1 = max([len(path) for _, path in file_dict.items()])
            
                    for f, path in file_dict.items():
                        exists = 'found' if os.path.isfile('data/' + path) else 'not found'
                        print (('{:<%i} {:<%i} {:<10}'%(align_0+4, align_1+4)).format(f, path, exists))
        
        
        elif sep[0] == 'clear_buckets': #empty the current bucket list
            param_dict['buckets'] = []


        elif sep[0] == 'set_file': #set the name or path of a file
            if len(sep)!= 3:
                print('set_file takes two arguments, the first is which file to set, the second is the path to the file, the path must be relative to the directory that has been mounted to /data')
            if os.path.isfile('data/' + sep[2]):
                if sep[1] not in file_dict.keys():
                    print('file id not recognised, use list files to see which files can be set')
                else:
                    file_dict[sep[1]] = sep[2]
                    file_dict_changes[sep[1]] = True
            else:
                cont = input('file not found, set anyway (y,n)')
                if cont == 'y' or cont == 'Y':
                    file_dict[sep[1]] = sep[2]
                    file_dict_changes[sep[1]] = True
                    
            
        elif sep[0] == 'set':#set the value of a parameter
            if len(sep) != 3:
                print('set takes two arguments, the first is the parameter to set, the second is the value')
            else:
                param_dict[sep[1]] = eval(sep[2])
                print(sep[1] + ' = ' + sep[2])
                
                if sep[1] == 'input_symbols':
                    if not file_dict_changes['source_ids']:
                        file_dict['source_ids'] = 'giga-fren.release2.fixed.en.ids%i'%param_dict['input_symbols']
                    if not file_dict_changes['source_vocab']:
                        file_dict['source_vocab'] = 'vocab%i.from'%param_dict['input_symbols']
                
                if sep[1] == 'output_symbols':
                    if not file_dict_changes['target_ids']:
                        file_dict['target_ids'] = 'giga-fren.release2.fixed.fr.ids%i'%param_dict['output_symbols']
                    if not file_dict_changes['target_vocab']:
                        file_dict['target_vocab'] = 'vocab%i.to'%param_dict['output_symbols']
                        
        
        elif sep[0] == 'add_bucket': #add a new bucket to the bucket list
            if len(sep) != 2:
                print('add_bucket takes one argument, the new bucket to add, this should be a tuple')
            else:
                param_dict['buckets'].append(eval(sep[1]))
    
    
        elif sep[0] == 'show': #show the value of a parameter
            if len(sep) != 2:
                print('show takes one argument, the name of the parameter to show')
            else:
                try:
                    print(param_dict[sep[1]])
                except:
                    print('parameter not found')
                
                
        elif sep[0] == 'help': #print help string
            print(help_string)
        
        
        elif sep[0] == 'get_data': #download and prepare data files
        
            #check which files already exist
            found = {}
            for f, path in file_dict.items():
                found[f] = os.path.isfile('data/' + path)
            
            #check if all the necessary files exist
            if found['source_ids'] and found['target_ids'] and found['source_vocab'] and found['target_vocab'] and found['test_source_ids'] and found['test_target_ids']:
                print('found all necessary files')
            
            else:
                print_dict(found)
                cont = input("Download and prepare necessary files? This may take several hours (y,n)")
                if cont == 'y' or cont == 'Y':
                    if found['unzipped_source'] and found['unzipped_target']:
                        print('unzipped data found, preparing data')
                        prepare_data('/data', file_dict['unzipped_source'], file_dict['unzipped_target'], param_dict['input_symbols'], param_dict['output_symbols'])
                    else:
                        prepare_wmt_data('/data', param_dict['input_symbols'], param_dict['output_symbols'])
        
        
        elif sep[0] == 'fit': #fit the model using current parameters and data files
            print('fitting')
            
            param_dict['buckets'] = sort_buckets(param_dict['buckets'])
            qr_list = [] #effectively used as a pointer to the queue runner
            input_fn = make_input_fn(param_dict['batch_size'],
                                     file_dict['source_ids'], 
                                     file_dict['target_ids'], 
                                     param_dict['max_samples_to_load'], 
                                     param_dict['buckets'],
                                     qr_list)
                                        
            param_dict['qr'] = qr_list
            
            estimator = tf.contrib.learn.Estimator(model_fn, params = param_dict, config = run_config)
            estimator.fit(input_fn = input_fn, steps = param_dict['steps'])
            
        
        
        elif sep[0] == 'eval': #evaluate model using current parameters and data files
            #almost the same as fit, except only take 1 step. TODO should suppress the training ops here
            param_dict['buckets'] = sort_buckets(param_dict['buckets'])
            qr_list = []
            input_fn = make_input_fn(param_dict['batch_size'],
                                     file_dict['test_source_ids'], 
                                     file_dict['test_target_ids'], 
                                     param_dict['max_samples_to_load'], 
                                     param_dict['buckets'],
                                     qr_list)
                                        
            param_dict['qr'] = qr_list
            estimator = tf.contrib.learn.Estimator(model_fn, params = param_dict, config = run_config)
            results = estimator.evaluate(input_fn = input_fn, steps = 1)
                
            print_dict(results)


        elif sep[0] == 'decode': #translate a phrase using the current model
            
            phrase = sep[1:]
            #load source and target vocab if they do not already exist
            try:
                source_vocab
            except:
                source_vocab, rev_source_vocab = initialize_vocabulary('data/' + file_dict['source_vocab'])
             
            try:
                target_vocab
            except:
                target_vocab, rev_target_vocab = initialize_vocabulary('data/' + file_dict['target_vocab'])
            
            #convert the given phrase into tokens
            phrase_tokens = []
            for word in phrase:
                try:
                    phrase_tokens.append(source_vocab[word.encode('UTF-8')])
                except:
                    print('unknown word %s'%word)
                    phrase_tokens.append(UNK)
            phrase_tokens.append(EOS)
            
            #create an estimator with the correct sized bucket
            param_temp = copy(param_dict)
            bucket = (len(phrase_tokens), len(phrase_tokens)+5)
            param_temp['buckets'] = [bucket]
            
            decoding_estimator = tf.contrib.learn.Estimator(model_fn, params = param_temp, config = run_config)
            
            #make the input function, the decoder inputs will not be used so they can all be PAD
            def pred_input_fn():
                x_dict = {0: tf.convert_to_tensor(np.reshape(phrase_tokens, [1,bucket[0],1]), dtype = tf.int32),
                          1: tf.convert_to_tensor(np.reshape([PAD]*bucket[1], [1,bucket[1],1]), dtype = tf.int32)}
                
                labels_dict = None
                return (x_dict, labels_dict)

            #make the prediction
            outputs = decoding_estimator.predict(input_fn = pred_input_fn)
            
            #only one prediction is made, so get the first prediction
            outputs = outputs.send(None)
            
            #convert the tokens into words
            outputs = outputs[bucket]
            words = []
            for token in outputs:
                if token == EOS:
                    break

                words.append(rev_target_vocab[token].decode('UTF-8'))
            translation = ' '.join(words)

            print(' '.join(phrase) + '  -->  ' + translation)

    
        elif sep[0] == 'look_up':
            #load vocab if it does not already exist
            try:
                source_vocab
            except:
                source_vocab, rev_source_vocab = initialize_vocabulary('data/' + file_dict['source_vocab'])
             
            try:
                target_vocab
            except:
                target_vocab, rev_target_vocab = initialize_vocabulary('data/' + file_dict['target_vocab'])
                
            if len(sep)!= 2:
                print ("look_up takes one argument, the token to loop_up")
            else:
                print("source: %s, target: %s"%(rev_source_vocab[eval(sep[1])], rev_target_vocab[eval(sep[1])]))
            
        #unknown command error
        else:
            print("Unknown command %s"%sep[0])
