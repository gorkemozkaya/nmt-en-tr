import tensorflow as tf

from tensor2tensor.utils import trainer_lib

# Set a seed so that we have deterministic outputs.
RANDOM_SEED = 301
trainer_lib.set_random_seed(RANDOM_SEED)


import os


# `Problem` is the base class for any dataset that we want to add to T2T -- it
# unifies the specification of the problem for generating training data,
# training, evaluation and inference.
#
# All its methods (except `generate_data`) have reasonable default
# implementations.
#
# A sub-class must implement `generate_data(data_dir, tmp_dir)` -- this method
# is called by t2t-trainer or t2t-datagen to actually generate TFRecord dataset
# files on disk.
from tensor2tensor.data_generators import problem

# Certain categories of problems are very common, like where either the input or
# output is text, for such problems we define an (abstract) sub-class of
# `Problem` called `Text2TextProblem` -- this implements `generate_data` in
# terms of another function `generate_samples`. Sub-classes must override
# `generate_samples` and `is_generate_per_split`.
from tensor2tensor.data_generators import text_problems

# Every non-abstract problem sub-class (as well as models and hyperparameter
# sets) must be registered with T2T so that T2T knows about it and can look it
# up when you specify your problem on the commandline to t2t-trainer or
# t2t-datagen.
#
# One uses:
# `register_problem` for a new Problem sub-class.
# `register_model` for a new T2TModel sub-class.
# `register_hparams` for a new hyperparameter set. All hyperparameter sets
# typically extend `common_hparams.basic_params1` (directly or indirectly).
from tensor2tensor.utils import registry


# By default, when you register a problem (or model or hyperparameter set) the
# name with which it gets registered is the 'snake case' version -- so here
# the Problem class `SortWordsAccordingToLengthRandom` will be registered with
# the name `sort_words_according_to_length_random`.
#
# One can override this default by actually assigning a name as follows:
# `@registry.register_problem("my_awesome_problem")`
#
# The registered name is specified to the t2t-trainer or t2t-datagen using the
# commandline flag `--problem`.
@registry.register_problem

# We inherit from `Text2TextProblem` which takes care of a lot of details
# regarding reading and writing the data to disk, what vocabulary type one
# should use, its size etc -- so that we need not worry about them, one can,
# of course, override those.
class TranslateEnTr(text_problems.Text2TextProblem):
  """Translate English to Turkish"""

  # START: Methods we should override.

  # The methods that need to be overriden from `Text2TextProblem` are:
  # `is_generate_per_split` and
  # `generate_samples`.

  @property
  def is_generate_per_split(self):
    # If we have pre-existing data splits for (train, eval, test) then we set
    # this to True, which will have generate_samples be called for each of the
    # dataset_splits.
    #
    # If we do not have pre-existing data splits, we set this to False, which
    # will have generate_samples be called just once and the Problem will
    # automatically partition the data into dataset_splits.
    return False

  def generate_samples(self, data_dir, tmp_dir, dataset_split):

    import re
    re0 = re.compile('\w.*|$')

    def preprocess(x):
      """
      Remove the non word characters at the beginning of a sentence.
      :param x: Input string
      :return: cleaned version of the string
      """
      return re0.search(x).group()

    en_tr_corpus_dir = os.environ["EN_TR_CORPUS_DIR"]

    # PART 1 - OPEN SUBTITLES CORPUS
    with open(en_tr_corpus_dir + 'OpenSubtitles.en-tr.en') as f_en, open(
            en_tr_corpus_dir + 'OpenSubtitles.en-tr.tr') as f_tr:
        data_iterator = zip(f_en, f_tr)
        t = 0
        for sentence_input, sentence_target in data_iterator:
            t += 1
            source = preprocess(sentence_input)
            target = preprocess(sentence_target)
            if t % 10 == 0:  # downsampling
                yield {
                    "inputs": source,
                    "targets": target,
                }
    # PART 2 - NEWS ARTICLES CORPUS
    with open(en_tr_corpus_dir + 'SETIMES2.en-tr.en') as f_en, open(en_tr_corpus_dir + 'SETIMES2.en-tr.tr') as f_tr:
        data_iterator = zip(f_en, f_tr)
        for sentence_input, sentence_target in data_iterator:
            source = preprocess(sentence_input)
            target = preprocess(sentence_target)

            yield {
              "inputs"  : source,
              "targets" : target,
            }


    # PART 3 - TED TALKS
    with open(en_tr_corpus_dir + 'TED2013.en-tr.en') as f_en, open(en_tr_corpus_dir + 'TED2013.en-tr.tr') as f_tr:
        data_iterator = zip(f_en, f_tr)
        for sentence_input, sentence_target in data_iterator:
            source = preprocess(sentence_input)
            target = preprocess(sentence_target)

            yield {
              "inputs"  : source,
              "targets" : target,
            }

    # PART 4 - BIANET
    with open(en_tr_corpus_dir + 'bianet-entr-en.txt') as f_en, open(en_tr_corpus_dir + 'bianet-entr-tr.txt') as f_tr:
        data_iterator = zip(f_en, f_tr)
        for sentence_input, sentence_target in data_iterator:
            source = preprocess(sentence_input)
            target = preprocess(sentence_target)

            yield {
              "inputs"  : source,
              "targets" : target,
            }


  # END: Methods we should override.

  # START: Overridable methods.

  @property
  def vocab_type(self):
    # We can use different types of vocabularies, `VocabType.CHARACTER`,
    # `VocabType.SUBWORD` and `VocabType.TOKEN`.
    #
    # SUBWORD and CHARACTER are fully invertible -- but SUBWORD provides a good
    # tradeoff between CHARACTER and TOKEN.
    return text_problems.VocabType.SUBWORD

  @property
  def approx_vocab_size(self):
    # Approximate vocab size to generate. Only for VocabType.SUBWORD.
    return 2**14  # ~16k

  @property
  def dataset_splits(self):
    # Since we are responsible for generating the dataset splits, we override
    # `Text2TextProblem.dataset_splits` to specify that we intend to keep
    # 80% data for training and 10% for evaluation and testing each.
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 8,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  @property
  def max_subtoken_length(self):
      """Maximum subtoken length when generating vocab.
      SubwordTextEncoder vocabulary building is quadratic-time wrt this variable,
      setting it to None uses the length of the longest token in the corpus.
      Returns:
        an integer or None
      """
      return 8

 # END: Overridable methods.


@registry.register_problem

# We inherit from `Text2TextProblem` which takes care of a lot of details
# regarding reading and writing the data to disk, what vocabulary type one
# should use, its size etc -- so that we need not worry about them, one can,
# of course, override those.
class TranslateTrEn(text_problems.Text2TextProblem):
  """Translate English to Turkish"""

  # START: Methods we should override.

  # The methods that need to be overriden from `Text2TextProblem` are:
  # `is_generate_per_split` and
  # `generate_samples`.

  @property
  def is_generate_per_split(self):
    # If we have pre-existing data splits for (train, eval, test) then we set
    # this to True, which will have generate_samples be called for each of the
    # dataset_splits.
    #
    # If we do not have pre-existing data splits, we set this to False, which
    # will have generate_samples be called just once and the Problem will
    # automatically partition the data into dataset_splits.
    return False

  def generate_samples(self, data_dir, tmp_dir, dataset_split):

    import re
    re0 = re.compile('\w.*|$')

    def preprocess(x):
      """
      Remove the non word characters at the beginning of a sentence.
      :param x: Input string
      :return: cleaned version of the string
      """
      return re0.search(x).group()

    en_tr_corpus_dir = os.environ["EN_TR_CORPUS_DIR"]

    # PART 1 - OPEN SUBTITLES CORPUS
    with open(en_tr_corpus_dir + 'OpenSubtitles.en-tr.en') as f_en, open(
            en_tr_corpus_dir + 'OpenSubtitles.en-tr.tr') as f_tr:
        data_iterator = zip(f_tr, f_en)
        t = 0
        for sentence_input, sentence_target in data_iterator:
            t += 1
            source = preprocess(sentence_input)
            target = preprocess(sentence_target)
            if t % 50 == 0:  # downsampling
                yield {
                    "inputs": source,
                    "targets": target,
                }
    # PART 2 - NEWS ARTICLES CORPUS
    with open(en_tr_corpus_dir + 'SETIMES2.en-tr.en') as f_en, open(en_tr_corpus_dir + 'SETIMES2.en-tr.tr') as f_tr:
        data_iterator = zip(f_tr, f_en)
        for sentence_input, sentence_target in data_iterator:
            source = preprocess(sentence_input)
            target = preprocess(sentence_target)

            yield {
              "inputs"  : source,
              "targets" : target,
            }



  # END: Methods we should override.

  # START: Overridable methods.

  @property
  def vocab_type(self):
    # We can use different types of vocabularies, `VocabType.CHARACTER`,
    # `VocabType.SUBWORD` and `VocabType.TOKEN`.
    #
    # SUBWORD and CHARACTER are fully invertible -- but SUBWORD provides a good
    # tradeoff between CHARACTER and TOKEN.
    return text_problems.VocabType.SUBWORD

  @property
  def approx_vocab_size(self):
    # Approximate vocab size to generate. Only for VocabType.SUBWORD.
    return 2**14  # ~16k

  @property
  def dataset_splits(self):
    # Since we are responsible for generating the dataset splits, we override
    # `Text2TextProblem.dataset_splits` to specify that we intend to keep
    # 80% data for training and 10% for evaluation and testing each.
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 8,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }, {
        "split": problem.DatasetSplit.TEST,
        "shards": 1,
    }]

  @property
  def max_subtoken_length(self):
      """Maximum subtoken length when generating vocab.
      SubwordTextEncoder vocabulary building is quadratic-time wrt this variable,
      setting it to None uses the length of the longest token in the corpus.
      Returns:
        an integer or None
      """
      return 8

 # END: Overridable methods.
