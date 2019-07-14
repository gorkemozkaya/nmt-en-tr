import os, glob
from  tensor2tensor.data_generators.text_problems import Text2TextProblem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
import tensorflow.contrib.eager as tfe
from tensor2tensor.utils import trainer_lib


hparams = trainer_lib.create_hparams(hparams_set, data_dir=data_dir, problem_name="translation_problem")


assert tfe.executing_eagerly() == True, "Eager execution needs to be enabled for this module."

@registry.register_problem
# We inherit from `Text2TextProblem`
class TranslationProblem(Text2TextProblem):
    pass

# Enable TF Eager execution
import tensorflow as tf

class TranslationModel:
    def __init__(self, model_path):
        t2tproblem = TranslationProblem()
        data_dir = os.path.join(model_path, 'data')
        ckpt_dir = os.path.join(model_path, 'model')
        try:
            vocab_filepath = glob(os.path.join(data_dir, '*subwords')[0]
            encoder = text_encoder.SubwordTextEncoder(vocab_filepath)
            ckpt_path = tf.train.latest_checkpoint(ckpt_dir)
        except Exception as e:
            print(e)

