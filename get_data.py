
import gzip
import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf
from random import choice, random


# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

PAD = 0
GO = 1
EOS = 2
UNK = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

# URLs for WMT data.
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"


def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not os.path.exists(directory):
    print("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not os.path.exists(filepath):
    print("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Successfully downloaded", filename, statinfo.st_size, "bytes")
  return filepath


def gunzip_file(gz_path, new_path):
  """Unzips from gz_path into new_path."""
  print("Unpacking %s to %s" % (gz_path, new_path))
  with gzip.open(gz_path, "rb") as gz_file:
    with open(new_path, "wb") as new_file:
      for line in gz_file:
        new_file.write(line)


def get_wmt_enfr_train_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  train_path = os.path.join(directory, "giga-fren.release2.fixed")
  if not (gfile.Exists(train_path +".fr") and gfile.Exists(train_path +".en")):
    corpus_file = maybe_download(directory, "training-giga-fren.tar",
                                 _WMT_ENFR_TRAIN_URL)
    print("Extracting tar file %s" % corpus_file)
    with tarfile.open(corpus_file, "r") as corpus_tar:
      def is_within_directory(directory, target):
          
          abs_directory = os.path.abspath(directory)
          abs_target = os.path.abspath(target)
      
          prefix = os.path.commonprefix([abs_directory, abs_target])
          
          return prefix == abs_directory
      
      def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
      
          for member in tar.getmembers():
              member_path = os.path.join(path, member.name)
              if not is_within_directory(path, member_path):
                  raise Exception("Attempted Path Traversal in Tar File")
      
          tar.extractall(path, members, numeric_owner=numeric_owner) 
          
      
      safe_extract(corpus_tar, directory)
    gunzip_file(train_path + ".fr.gz", train_path + ".fr")
    gunzip_file(train_path + ".en.gz", train_path + ".en")
  return train_path


def get_wmt_enfr_dev_set(directory):
  """Download the WMT en-fr training corpus to directory unless it's there."""
  dev_name = "newstest2013"
  dev_path = os.path.join(directory, dev_name)
  if not (gfile.Exists(dev_path + ".fr") and gfile.Exists(dev_path + ".en")):
    dev_file = maybe_download(directory, "dev-v2.tgz", _WMT_ENFR_DEV_URL)
    print("Extracting tgz file %s" % dev_file)
    with tarfile.open(dev_file, "r:gz") as dev_tar:
      fr_dev_file = dev_tar.getmember("dev/" + dev_name + ".fr")
      en_dev_file = dev_tar.getmember("dev/" + dev_name + ".en")
      fr_dev_file.name = dev_name + ".fr"  # Extract without "dev/" prefix.
      en_dev_file.name = dev_name + ".en"
      dev_tar.extract(fr_dev_file, directory)
      dev_tar.extract(en_dev_file, directory)
  return dev_path


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_bytes(line)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                            tokenizer, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_wmt_data(data_dir, en_vocabulary_size, fr_vocabulary_size, tokenizer=None):
  """Get WMT data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for English training data-set,
      (2) path to the token-ids for French training data-set,
      (3) path to the token-ids for English development data-set,
      (4) path to the token-ids for French development data-set,
      (5) path to the English vocabulary file,
      (6) path to the French vocabulary file.
  """
  # Get wmt data to the specified directory.
  train_path = get_wmt_enfr_train_set(data_dir)


  from_train_path = train_path + ".en"
  to_train_path = train_path + ".fr"
  return prepare_data(data_dir, from_train_path, to_train_path, en_vocabulary_size,
                      fr_vocabulary_size, tokenizer)


def prepare_data(data_dir, from_train_path, to_train_path, from_vocabulary_size,
                 to_vocabulary_size, tokenizer=None):
  """Preapre all necessary files that are required for the training.

    Args:
      data_dir: directory in which the data sets will be stored.
      from_train_path: path to the file that includes "from" training samples.
      to_train_path: path to the file that includes "to" training samples.
      from_dev_path: path to the file that includes "from" dev samples.
      to_dev_path: path to the file that includes "to" dev samples.
      from_vocabulary_size: size of the "from language" vocabulary to create and use.
      to_vocabulary_size: size of the "to language" vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for "from language" training data-set,
        (2) path to the token-ids for "to language" training data-set,
        (3) path to the token-ids for "from language" development data-set,
        (4) path to the token-ids for "to language" development data-set,
        (5) path to the "from language" vocabulary file,
        (6) path to the "to language" vocabulary file.
    """
  # Create vocabularies of the appropriate sizes.
  to_vocab_path = os.path.join(data_dir, "vocab%d.to" % to_vocabulary_size)
  from_vocab_path = os.path.join(data_dir, "vocab%d.from" % from_vocabulary_size)
  create_vocabulary(to_vocab_path, to_train_path , to_vocabulary_size, tokenizer)
  create_vocabulary(from_vocab_path, from_train_path , from_vocabulary_size, tokenizer)

  # Create token ids for the training data.
  to_train_ids_path = to_train_path + (".ids%d" % to_vocabulary_size)
  from_train_ids_path = from_train_path + (".ids%d" % from_vocabulary_size)
  data_to_token_ids(to_train_path, to_train_ids_path, to_vocab_path, tokenizer)
  data_to_token_ids(from_train_path, from_train_ids_path, from_vocab_path, tokenizer)


  return (from_train_ids_path, to_train_ids_path,
          from_vocab_path, to_vocab_path)




def pad(data, target_length):
    """ pads data with the PAD token up to a given length
        Args:
            data: data to be padded
            target_length: the length of data after being padded
        Raises:
            ValueError: if data is longer than target length
    """
    if len(data)>target_length:
        raise ValueError("data is longer than target length, lengths: {}, {}".format(len(data), target_length))
        
    data = data + [PAD]*(target_length - len(data))
    return data

#not used, reimplemented in task_1_neat using queue runners

# def get_data(source_path, target_path, train_num_sample, test_num_sample, buckets):
#     """ get num_sample samples from source path and target path then split it into buckets.
#         This function samples from the data file, as the file is very large it is impractical to load
#         it all at once so batches are taken, these batches are then sampled from further by the input_fn.
#         
#         The difficulty here is that the input_fn has to re-defined each time a new batch from the file
#         is required, this also means re-calling the estimator.fit function which has some unwanted side
#         effects in regards to saving logs and model parameters.
#         
#         It would be ideal for this function to return a queue which samples from the file when it is
#         below a certain fullness. However, as the file needs to be processed heavily to fit samples into
#         buckets and PAD those samples, something that would not be simple to do using a tensorflow graph
#         Args:
#             source_path: path to source language data
#             targte_path: path to target language data
#             train_num_sumple: number of training samples to get
#             test_num_sample: number of test samples to get
#             buckets: buckets into which the data will be placed based on length. The smallest possible
#                     bucket is used
#         
#     """
#     
#     train_data = [ []  for _ in buckets]
#     test_data = [ []  for _ in buckets]
#     with gfile.GFile(source_path, mode="r") as source_file:
#         with gfile.GFile(target_path, mode="r") as target_file:
#         
#             
#             source, target = source_file.readline(), target_file.readline()
#             i = 0
#             total_bucket_sizes = [s + t for s,t in buckets]
#             
#             #sort buckets by total size to make it easier to find the smallest bucket that will fit a data sample
#             enum_sorted_buckets = [bucket for _, bucket in sorted(zip(total_bucket_sizes, enumerate(buckets)), key = lambda pair: pair[0])]
#             
#             #find the number of lines in the file and store this number on the disk
#             if os.path.isfile('data/num_lines'):
#               with open('data/num_lines', 'r') as f:
#                 num_lines = int(f.read())
#               
#             else:
#               nun_lines = 0
#               
#               while source:
#                 source = source_file.readline()
#                 num_lines += 1
#                 
#               source_file.seek(0)
#               source = source_file.readline()
#               
#               with open('data/num_lines', 'w') as f:
#                 f.write(str(num_lines))
#             
#             #the first test_num_sample lines in the file make up the test set, first read those line
#             #and then sample from the remaining file to get the train set
#             i = 0
#             while i < test_num_sample:
#                 
#                 source = [int(x) for x in source.split()] + [EOS]
#                 target = [int(x) for x in target.split()] + [EOS]
#                 
#                 s_len = len(source)
#                 t_len = len(target)
#                 bucket_id = 0
#                 
#                 
#                 
#                 for bucket_id, bucket in enum_sorted_buckets:
#                     if s_len < bucket[0] and t_len < bucket[1]:
#                         test_data[bucket_id].append((pad(source, bucket[0]), pad(target, bucket[1])))
#                         i+=1
#                         break
#                 
#                 source, target = source_file.readline(), target_file.readline()    
#                 if not source or not target:
#                     source_file.seek(0)
#                     target_file.seek(0)
#                     source, target = source_file.readline(), target_file.readline() 
#             
#             #sample from the file
#             while i < train_num_sample:
#                 
#                 source = [int(x) for x in source.split()] + [EOS]
#                 target = [int(x) for x in target.split()] + [EOS]
#                 
#                 s_len = len(source)
#                 t_len = len(target)
#                 bucket_id = 0
#                 
#                 for bucket_id, bucket in enum_sorted_buckets:
#                     if s_len < bucket[0] and t_len < bucket[1]:
#                           train_data[bucket_id].append((pad(source, bucket[0]), pad(target, bucket[1])))
#                           i += 1
#                         break
#                 
#                 source, target = source_file.readline(), target_file.readline()
#                 if not source or not target:
#                     source_file.seek(0)
#                     target_file.seek(0)
#                     source, target = source_file.readline(), target_file.readline()
#                   
#              
#             
#     #if a bucket is empty fill it with PAD, this will be ignored by the trainer and evaluator but makes data handling easier
#     for bucket_id, bucket in enum_sorted_buckets:
#         if len(train_data[bucket_id]) == 0:
#             train_data[bucket_id].append(( pad([PAD], bucket[0]), pad([PAD], bucket[1])))
#         if len(test_data[bucket_id]) == 0:
#             test_data[bucket_id].append(( pad([PAD], bucket[0]), pad([PAD], bucket[1])))   
#         
#     
#     return train_data, test_data