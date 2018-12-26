import os
import re
import tensorflow as tf
from tensorflow.python.platform import gfile
from utils.conf import gen_config as conf
# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


VOCABULARY_SIZE = conf.vocab_size
DATA_DIR = conf.data_dir
VOCABULARY_PATH = (DATA_DIR + "vocab%d.all" % VOCABULARY_SIZE)
# NAMESET_PATH = 'english_names_corpus.txt'

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(r"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

def initialize_nameset(nameset_path):
    if gfile.Exists(nameset_path):
        rev_vocab = []
        with gfile.GFile(nameset_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        nameset = set(i for i in rev_vocab)
        return nameset
    else:
        raise ValueError("nameset file %s not found.", nameset_path)
# Regular expressions used to tokenize.
# _NAME_SET = initialize_nameset(NAMESET_PATH)

def create_vocabulary(vocabulary_path, data_path_list, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True, normalize_name=True):
  """Create vocabulary file (if it does not exist yet) from disc_data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: disc_data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each disc_data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
    normalize_name: Boolean; if true, all name are replaced by _Name >
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from disc_data %s" % (vocabulary_path, data_path_list))
    vocab = {}
    for data_path in data_path_list:
        with gfile.GFile(data_path, mode="r") as f:
          counter = 0
          for line in f:
            counter += 1
            if counter % 100000 == 0:
              print("  processing line %d" % counter)
            line = tf.compat.as_str_any(line)
            tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
            for w in tokens:
              word = _DIGIT_RE.sub("0", w) if normalize_digits else w
              #if normalize_name:
                  #word = '_NAME' if w in _NAME_SET else w
              if word in vocab:
                vocab[word] += 1
              else:
                vocab[word] = 1

    vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocabulary_size:
      vocab_list = vocab_list[:max_vocabulary_size]
    with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
      for w in vocab_list:
        vocab_file.write(w + "\n")

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
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    if type(space_separated_fragment) == bytes:
      space_separated_fragment = space_separated_fragment.decode()
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]

def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True, normalize_name=False):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0.

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
  return [vocabulary.get(_DIGIT_RE.sub("0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary,
                      tokenizer=None, normalize_digits=True):
  """Tokenize disc_data file and turn into token-ids using given vocabulary file.

  This function loads disc_data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the disc_data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing disc_data in %s" % data_path)
    #print("target path: ", target_path)
    #vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocabulary, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def prepare_chitchat_data(data_dir, vocabulary, vocabulary_size, tokenizer=None):
    """
    """
    # train_path = get_wmt_enfr_train_set(data_dir)
    train_path = os.path.join(data_dir, "train")
    answer_train_ids_path = train_path + (".ids%d.answer" % vocabulary_size)
    query_train_ids_path = train_path + (".ids%d.query" % vocabulary_size)
    data_to_token_ids(train_path + ".dec", answer_train_ids_path, vocabulary, tokenizer)
    data_to_token_ids(train_path + ".enc", query_train_ids_path, vocabulary, tokenizer)
    return (query_train_ids_path, answer_train_ids_path)

def prepare_data_dirlist(data_dir):
    """
    """
    f1 = data_dir + "train.enc"
    f2 = data_dir + "train.dec"
    f3 = data_dir + "test.enc"
    f4 = data_dir + "test.dec"
    list = [f1, f2, f3, f4]
    return list

if __name__ == '__main__':
    #main function for test
    DATA_DIR_LIST = prepare_data_dirlist(DATA_DIR)
    create_vocabulary(VOCABULARY_PATH, DATA_DIR_LIST, VOCABULARY_SIZE)
    vocab ,_ = initialize_vocabulary(VOCABULARY_PATH)
    prepare_chitchat_data(DATA_DIR, vocab, VOCABULARY_SIZE)
