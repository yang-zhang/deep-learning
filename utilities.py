from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

def plot_keras_model(model, show_shapes=True, show_layer_names=True):
    return SVG(model_to_dot(model, 
                            show_shapes=show_shapes, 
                            show_layer_names=show_layer_names).create(prog='dot', format='svg'))