import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tf_keras

model = tf_keras.models.load_model("keras_model.h5", compile=False)
model.summary()
print("Input shape:", model.input_shape)


print("hello 5")