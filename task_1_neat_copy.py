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
Commands can be run automatically at startup by creating a file called "com" in the data directory. Each line of this\n
file should be an instance of one of the commands above, each line of "com" will be run in order. More commands can\n
be entered after the "com" file is finished.

                """
    
    
def model_fn(features, labels, mode, params, config):
    """model function to be passes to tf.contrib.learn.Estimator
    
    Args:
        features: dictionary of feature tensors 
        labels: dictionary of label tensors
        mode: mode that the estimator is being run in (train, evaluate or infer)
        params: dictionary of model hyper-parameters
        config: configuration object, will receive what is passed to Estimator in the config argument
    
    returns: ModelFnOps object
    
    
    """
    buckets = params['buckets']
    
    if mode is tf.contrib.learn.ModeKeys.TRAIN:
        params['is_training'] = tf.constant(True, dtype = tf.bool)
        params['keep_prob'] = params['keep_prob']
    else:
        params['is_training'] = tf.constant(False, dtype = tf.bool)
        params['keep_prob'] = 1.0
    
    with tf.variable_scope('model'):
        token_prediction, loss, num_non_zero, step = create_model(features,
                                                            labels,
                                                            params,
                                                            mode)
    if mode != tf.contrib.learn.ModeKeys.INFER:
        
        total_loss = sum(loss)/tf.cast(sum(num_non_zero),tf.float32)
        loss = [l/(tf.cast(n, tf.float32) + 0.0001) for l,n in zip(loss, num_non_zero)]
        
        step = tf.train.get_global_step()
        inc_step = tf.assign_add(step,1)
        
        alpha = tf.Variable(params['alpha'])
        
        #previous losses
        p_losses = tf.Variable(tf.zeros([50], tf.float32))
        new_alpha = tf.cond(tf.logical_and(step>10, tf.reduce_max(p_losses) < total_loss),lambda: alpha*params['alpha_decay'], lambda: alpha)
        update_alpha = tf.assign(alpha, new_alpha)
        with tf.control_dependencies([update_alpha]): #ensure that alpha updates before p_losses
            update_p_losses = tf.assign(p_losses, tf.concat([tf.slice(p_losses, [1], [-1]), tf.expand_dims(total_loss,0)], 0))
        
        
        
        
        
        with tf.variable_scope('training'): 
            opt= tf.train.AdamOptimizer(alpha)
            grads_vars = opt.compute_gradients(total_loss)
            clipped_grads, _ = tf.clip_by_global_norm([a for a,b in grads_vars], params['max_grad_norm'])
            clipped_grads_vars = [(grad, var) for grad,var in zip(clipped_grads, [b for a,b in grads_vars])]
            with tf.control_dependencies([inc_step, update_p_losses]):
                train_op = opt.apply_gradients(clipped_grads_vars)
        
    
        
        loss_dict = {}
        loss_summaries = []
        
        for i,l,bucket in zip(range(len(buckets)),loss,buckets):
            name = 'bucket_{}_loss'.format(i)
            loss_dict[name] = l
            loss_summaries.append(tf.summary.scalar(name, l))
            
        total_loss_summary = tf.summary.scalar('total_loss', total_loss)
        loss_dict['total_loss'] = total_loss
        
        alpha_summary = tf.summary.scalar('learning_rate', alpha)
    else:
        total_loss = None
        train_op = None
        loss_dict = None
    
    all_summaries = tf.summary.merge_all()
    
    pred_dict = {}
    for i,bucket in enumerate(buckets):
        pred_dict[bucket] = token_prediction[i]
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    sess = tf.Session()
    
    writer = tf.summary.FileWriter('logs', graph = sess.graph)
    
    scaffold = tf.train.Scaffold(saver = saver,
                                 init_op = init,
                                 summary_op = all_summaries)
                                 

    
    return tf.contrib.learn.ModelFnOps(mode = mode,
                                      predictions = pred_dict,
                                      loss = total_loss,
                                      train_op = train_op,
                                      scaffold = scaffold,
                                      eval_metric_ops = loss_dict
                                      )
                                      
def make_input_fn(batch_size, source_ids_file, target_ids_file, buckets, samples_to_load):
    """makes an input function to be passed to fit. Samples from data producing
    minibatches with size, batch_size. The data is split so that the relative
    population of each bucket doesn't change on average, so buckets with more samples
    in them are xmore likely to be sampled from.
    Args:
        batch_size: number of samples to get after each call of input_fn
        source_ids_file: 
    
    Returns:
        input_fn: function to be used with fit
    
    """
    train_data, test_data = get_data('data/' + source_ids_file,
    "data/" + target_ids_file, samples_to_load, batch_size, buckets)
    
    

    num_buckets = len(buckets)
    train_encoder_input = []
    train_decoder_input = []
    test_encoder_input = []
    test_decoder_input = []
    bucket_entries = []
    for i in range(num_buckets):
        train_encoder_input.append(np.asarray([a for a,b in train_data[i]]))
        train_decoder_input.append(np.asarray([b for a,b in train_data[i]]))
        
        test_encoder_input.append(np.asarray([a for a,b in test_data[i]]))
        test_decoder_input.append(np.asarray([b for a,b in test_data[i]]))
        
        bucket_entries.append(train_encoder_input[-1].shape[0])
    
    p = [n/sum(bucket_entries) for n in bucket_entries]

    def train_input_fn():
        
        features = {}
        labels = {}
        for i in range(num_buckets):
 
            total_batch_size = train_encoder_input[i].shape[0]
            bucket_batch_size = ceil(p[i]*batch_size)
        
            en_feat = tf.convert_to_tensor(np.expand_dims(train_encoder_input[i],2))
            label = tf.convert_to_tensor(train_decoder_input[i])
            
            train_decoder_input[i][train_decoder_input[i] == EOS] = PAD
            
            dec_feat = tf.convert_to_tensor(np.expand_dims(np.concatenate([GO*np.ones([total_batch_size,1]), train_decoder_input[i][:,:-1]],1), 2).astype(np.int64))
            
            tensors = [en_feat, label, dec_feat]
            
            batch_tensors = tf.train.shuffle_batch(tensors, bucket_batch_size, 3*bucket_batch_size, 2*bucket_batch_size, enqueue_many = True)
            
            features[i] = batch_tensors[0]
            features[i+num_buckets] = batch_tensors[2]
            labels[i] = tf.reshape(batch_tensors[1], [-1,1])
        
        return features, labels
        
    def test_input_fn():
        
        features = {}
        labels = {}
        for i in range(num_buckets):
 
            total_batch_size = test_encoder_input[i].shape[0]
            bucket_batch_size = ceil(p[i]*batch_size)
        
            en_feat = tf.convert_to_tensor(np.expand_dims(test_encoder_input[i],2))
            label = tf.convert_to_tensor(test_decoder_input[i])
            
            test_decoder_input[i][test_decoder_input[i] == EOS] = PAD
            
            dec_feat = tf.convert_to_tensor(np.expand_dims(np.concatenate([GO*np.ones([total_batch_size,1]), test_decoder_input[i][:,:-1]],1), 2).astype(np.int64))
            
            tensors = [en_feat, label, dec_feat]
            
            batch_tensors = tf.train.shuffle_batch(tensors, bucket_batch_size, 3*bucket_batch_size, 2*bucket_batch_size, enqueue_many = True)
            
            features[i] = batch_tensors[0]
            features[i+num_buckets] = batch_tensors[2]
            labels[i] = tf.reshape(batch_tensors[1], [-1,1])
        
        return features, labels    
    
    return train_input_fn, test_input_fn

def print_dict(dict):
    align = [0,0]
    for k, v in dict.items():
        if len(str(k)) > align[0]:
            align[0] = len(str(k))
        if len(str(v)) > align[1]:
            align[1] = len(str(v))
        
    for k, v in dict.items():
        print( ('{:<%i} {:<%i}'%(align[0]+4, align[1]+4)).format(str(k),str(v)))


if __name__ == '__main__':
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
                'samples_to_load': 3000}
                
    file_dict = {'source_ids': 'giga-fren.release2.fixed.en.ids%i'%param_dict['input_symbols'],
                 'target_ids': 'giga-fren.release2.fixed.fr.ids%i'%param_dict['output_symbols'],
                 'source_vocab': 'vocab%i.from'%param_dict['input_symbols'],
                 'target_vocab': 'vocab%i.to'%param_dict['output_symbols'],
                 'corpus_file': 'training-giga-fren.tar',
                 'unzipped_source': 'giga-fren.release2.fixed.en',
                 'unzipped_target': 'giga-fren.release2.fixed.fr'}
    
    file_dict_changes = {} #keep track of whether the four file names which depend on param_dict have been changed by the user
    for k in file_dict.keys():
        file_dict_changes[k] = False
    
    run_config = tf.contrib.learn.RunConfig(save_summary_steps = 10,
                                            model_dir = 'logs',
                                            save_checkpoints_secs = None,
                                            save_checkpoints_steps = 500)
                                            
    
    estimator = tf.contrib.learn.Estimator(model_fn, params = param_dict, config = run_config)
    
    print('use set param value to set parameters, list to list current parameters and values and run to start the trainer, type help for help')
    try:
        with open('data/com') as f:
            coms = f.read().splitlines()
    except:
        coms = []
    
    commands = 0
    while True:
        if len(coms) > commands:
            com = coms[commands]
        else:
            com = input(">>>  ")
        commands += 1
        sep = com.split()
        
        if sep[0] == 'list':
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
        
        elif sep[0] == 'clear_buckets':
            params_dict['buckets'] = []

        if sep[0] == 'set_file':
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
                
                
        if sep[0] == 'set':
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
                        
        
        if sep[0] == 'add_bucket':
            if len(sep) != 2:
                print('add_bucket takes one argument, the new bucket to add, this should be a tuple')
            else:
                param_dict['buckets'].append(eval(sep[1]))
    
        if sep[0] == 'show':
            if len(sep) != 2:
                print('show takes one argument, the name of the parameter to show')
            else:
                try:
                    print(param_dict[sep[1]])
                except:
                    print('parameter not found')
                    
        if sep[0] == 'help':
            print(help_string)
        
        if sep[0] == 'get_data':
            found = {}
            for f, path in file_dict.items():
                found[f] = os.path.isfile('data/' + path)
            
            if found['source_ids'] and found['target_ids'] and found['source_vocab'] and found['target_vocab']:
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
        
        
        if sep[0] == 'fit':
            print('fitting')
            estimator = tf.contrib.learn.Estimator(model_fn, params = param_dict, config = run_config)
            if len(sep) > 1:
                iters = eval(sep[1])
            else:
                iters = 1
            
            for _ in range(iters):
                train_input_fn, test_input_fn = make_input_fn(64, file_dict['source_ids'], file_dict['target_ids'], param_dict['buckets'], param_dict['samples_to_load'])
                estimator.fit(input_fn = train_input_fn, steps = param_dict['steps'])
        
        
        if sep[0] == 'eval':
            try:
                test_input_fn
            except:
                train_input_fn, test_input_fn = make_input_fn(64, file_dict['source_ids'], file_dict['target_ids'], param_dict['buckets'], param_dict['samples_to_load'])
            
            results = estimator.evaluate(input_fn = test_input_fn, steps = 1)
            print_dict(results)


        if sep[0] == 'decode':
            
            phrase = sep[1:]
            try:
                source_vocab
            except:
                source_vocab, rev_source_vocab = initialize_vocabulary('data/' + file_dict['source_vocab'])
             
            try:
                target_vocab
            except:
                target_vocab, rev_target_vocab = initialize_vocabulary('data/' + file_dict['target_vocab'])
            
            phrase_tokens = []
            for word in phrase:
                try:
                    phrase_tokens.append(source_vocab[word.encode('UTF-8')])
                except:
                    print('unknown word %s'%word)
                    phrase_tokens.append(UNK)
            phrase_tokens.append(EOS)
            
            param_temp = copy(param_dict)
            bucket = (len(phrase_tokens), len(phrase_tokens)+5)
            param_temp['buckets'] = [bucket]
            
            
            decoding_estimator = tf.contrib.learn.Estimator(model_fn, params = param_temp, config = run_config)
            def pred_input_fn():
                x_dict = {0: tf.convert_to_tensor(np.reshape(phrase_tokens, [1,bucket[0],1])),
                          1: tf.convert_to_tensor(np.reshape([0]*bucket[1], [1,bucket[1],1]))}
                
                labels_dict = None
                return (x_dict, labels_dict)

            outputs = decoding_estimator.predict(input_fn = pred_input_fn)
            outputs = outputs.send(None)
            
            outputs = outputs[bucket]
            words = []
            for token in outputs:
                if token == EOS:
                    break

                words.append(rev_target_vocab[token].decode('UTF-8'))
            translation = ' '.join(words)

            print(' '.join(phrase) + '  -->  ' + translation)

    
        if sep[0] == 'look_up':
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
