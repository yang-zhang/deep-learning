from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

def plot_keras_model(model, show_shapes=True, show_layer_names=True):
    return SVG(model_to_dot(model, 
                            show_shapes=show_shapes, 
                            show_layer_names=show_layer_names).create(prog='dot', format='svg'))

def print_weights_shape(model):
    weights = [i.get_weights() for i in model.layers]
    for i in weights:
        print('-' * 13)
        for j in i:
            print(j.shape)
