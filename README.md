## Synopsis

This project is a tensorflow implementation of a sequence to sequence (seq2seq) model, based on recurrent deep learning. The seq2seq model is used to learn to translate from a source language to a target language. By default the source langauage is english and the target is french.

## Installation

A docker image of this project is provided. The image contains two directories, logs/ and data/ . The logs directory is used to save model specific information, for example, saved parameters and tensorboard event files. The data directory is used for task specific information, for example, the data files used for the task.

To run the docker image use:

sudo docker run -it -v /host/directory/for/logs/model_name:/logs -v /host/directory/for/data:/data translate

-it runs the image in interactive mode, this is required for setting parameters and controlling fitting, evaluation and inference

-v mounts a host directory to the docker container

## Usage

when the docker image is run a console will appear, this console is running in python, use the command help for a list of commands.

## example

$ sudo docker run -it -v /host/directory/for/logs/model_name:/logs -v /host/directory/for/data:/data translate
use set param value to set parameters, list to list current parameters and values and run to start the trainer, type help for help

\> list params
 layer_size              300
num_layers              3
input_symbols           40000
output_symbols          40000 
embedding_size          300
attn_layer_size         30 
alpha                   0.001
alpha_decay             1.0
max_grad_norm           2.0
softmax_sample_size     500
keep_prob               0.5 
buckets                 \[(7, 10), (13, 20)]
steps                   1000 
task                    3 
max_samples_to_load     3000 
batch_size              64 
attn_method             nn 
l2_reg_strength         4e-05 
alpha_alpha             0.1 

\> set steps 3000
steps = 3000
\> show steps
3000
\> fit
fitting
\> eval
bucket_0_loss	5.54
bucket_1_loss	6.21
loss		5.74
global_step	3001

## files

task_1_neat.py: the main file in this project, contains the model_fn and input_fn used with the tensorflow estimator class. This file also handles user commands.

task_1_model.py: 
	create_model - defines the RNN cells for the encoder and decoder
	create_model_from_cells: manages the creation of an RNN from the raw cells

wrapper.py: custom wrappers and cells for the RNN. A wrapper can be applied to an existing cell, it then "wraps" the cell by processing the input and incoming state, calling the wrapped cell and processing the output and outgoing state.
	AttentionWrapper - A wrapper for an RNN cell which implements the attention mechanism defined in https://arxiv.org/pdf/1409.0473.pdf . This is a replacement for the tensorflow attention wrapper which claims to implement the same method but appears not to (https://github.com/tensorflow/tensorflow/issues/4427)
	DropoutWrapper - wrapper which applies dropout to the output of a cell. Does not apply dropout to the hidden state through time
	AttentionCell - An RNN cell which implements the attention mechanism defined in http://aclweb.org/anthology/D15-1166 . It is possible to use a cell rather than a wrapper for this mechanism as, unlike the method used in AttentionWrapper, is essentially feed forward.
	EmbeddingWrapper - wrapper which adds an embedding layer to the input of a cell. This is a copy of the tensorflow embedding wrapper, slightly modified to allow for multiple wrappings. 

rnns.py: contains function to create an RNN model from encoder and decoder cells
	find_attn_pos - recursively finds attention cell states in the full cell state tuple
	find_hidden_pos - recursively finds the hidden states of LSTM cells in the full cell state tuple
	MyRNN - creates the seq2seq rnn from encoder and decoder cells using dynamic_rnn for the encoder and raw_rnn for the decoder


get_data.py: functions for downloading and processesing training data, copied from the translate example
	maybe_download - check if it is necessary to download the data files and then download them if it is
	gunzip_file - unzips the downloaded data file
	get_wmt_enfr_train_set - downloads and unzips the training data
	get_wmt_enfr_dev_set - downloads and unzips the development data (maybe test data but I'm not sure if dev and train are from the same source)
	basic__tokenizer - converts the text in the unzipped data set into tokens, each token is an integer and each integer refers to a word
	create_vocabulary - creates a file containing a mapping from tokens to words
	initialize_vocabulary - loads the mapping created in create_vocabulary and returns a dictionary to go from words to token, and a list to go from tokens to words
	sentence_to_token_ids - converts a single sentence of words into tokens
	data_to_token_ids - converts a file containing many sentences of words into tokens
	prepare_wmt_data - fills a directory with the necessary data files
	prepare_data - creates vocab and id files from the unzipped data in a directory




#MachineTranslation
