# Text Segmentation

This folder contains everything regarding text segmentation. The core method,
`get_segmentation()` is provided by `Segmentation.py` and splits a text into
segments. `LightSegmentation.py` provides some methods that might be used outside
of `Segmentation.py` and don't require the RoBERTa model. This allows those methods
to be used without initializing the model, which may take a while.