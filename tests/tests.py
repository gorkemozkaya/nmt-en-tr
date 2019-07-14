import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from nmt_en_tr import TranslationModel



class BasicTests(test_util.TensorFlowTestCase):
  """Tests of the new, differentiable version of accumulate_n."""

  def testModelInitialization(self):
    TranslationModel('/tmp')
    # self.assertEqual(42, answer.numpy())

  # def testFloat(self):
  #   np.random.seed(12345)
  #   x = [np.random.random((1, 2, 3, 4, 5)) - 0.5 for _ in range(5)]
  #   tf_x = ops.convert_n_to_tensor(x)
  #   self.assertAllClose(sum(x), math_ops.accumulate_n(tf_x))
  #   self.assertAllClose(x[0] * 5,
  #                       math_ops.accumulate_n([tf_x[0]] * 5))
  #
  # def testGrad(self):
  #   np.random.seed(42)
  #   num_inputs = 3
  #   input_vars = [
  #       resource_variable_ops.ResourceVariable(10.0 * np.random.random(),
  #                                              name="t%d" % i)
  #       for i in range(0, num_inputs)
  #   ]
  #
  #   def fn(first, second, third):
  #     return math_ops.accumulate_n([first, second, third])
  #
  #   grad_fn = backprop.gradients_function(fn)
  #   grad = grad_fn(input_vars[0], input_vars[1], input_vars[2])
  #   self.assertAllEqual(np.repeat(1.0, num_inputs),  # d/dx (x + y + ...) = 1
  #                       [elem.numpy() for elem in grad])


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()