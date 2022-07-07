#to load the weights in tf2 for a model that is trained on tf1

reader = tf.train.load_checkpoint(ckpt_dir)
variable_map = {}
for var in tf.compat.v1.trainable_variables():
  var_name = var.name.split(":")[0]
  if reader.has_tensor(var_name):
    # tf.logging.info("Loading variable from checkpoint: %s", var_name)
    variable_map[var_name] = var
    print(var_name)
  else:
    tf.logging.info("Cannot find variable in checkpoint, skipping: %s",
                    var_name)
tf.compat.v1.train.init_from_checkpoint(ckpt_dir, variable_map)
