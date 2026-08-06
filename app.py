from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, groups=1, **kwargs):
        super().__init__(*args, **kwargs)

@st.cache_resource
def load_assets():
    model = load_model(
        "keras_model.h5",
        compile=False,
        custom_objects={"DepthwiseConv2D": PatchedDepthwiseConv2D},
    )
    with open("labels.txt", encoding="utf-8") as f:
        labels = [line.strip().split(" ", 1)[1] for line in f if line.strip()]
    return model, labels