from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, groups=1, **kwargs):
        super().__init__(*args, **kwargs)

model = load_model(
    "keras_model.h5",
    compile=False,
    custom_objects={"DepthwiseConv2D": PatchedDepthwiseConv2D},
)
model.summary()
print("Input shape:", model.input_shape)