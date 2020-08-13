from keras.models import load_model
from feedwork.utils.FileHelper import cat_path


def load_rnn_models(model_dir):
    """
    加载模型文件目录下的模型
    Args:
      model_dir: 模型文件所在目录

    Returns:
      模型的 encoder, decoder
    """
    encoder = load_model(cat_path(model_dir, 'encoder.h5'), compile=False)
    decoder = load_model(cat_path(model_dir, 'decoder.h5'), compile=False)
    return encoder, decoder


def load_cnn_model(model_dir):
    """
    加载模型文件目录下的模型
    Args:
      model_dir: 模型文件所在目录

    Returns:
      模型的 encoder, decoder
    """
    model = load_model(cat_path(model_dir, 'cnn_model.h5'), compile=False)
    return model
