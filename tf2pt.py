import re

def tf2pt_name(name_tf):
    param_type_dict = {
        'w': 'weight',
        'gamma': 'weight',
        'beta': 'bias',
        'moving_mean': 'running_mean',
        'moving_variance': 'running_var',
    }
    
    if name_tf.startswith('res_'):
        # res_2_0/bn_0/moving_variance
        pattern = (
            'res'
            r'_(\d)'  # 2 dil_idx
            r'_(\d)'  # 0 res_idx
            r'/(\w+)' # bn layer_type
            r'_(\d)'  # 0 layer_idx
            r'/(\w+)'  # moving_variance param_type
        )
        groups = re.match(pattern, name_tf).groups()
        dil_idx, res_idx, layer_type, layer_idx, param_type = groups
        param_idx = 3 if layer_type == 'conv' else 0
            
        name_pt = (
            f'block.{dil_idx}.dilation_block.{res_idx}.residual_block'
            f'.{layer_idx}.convolutional_block.{param_idx}.{param_type_dict[param_type]}'
        )
    elif name_tf.startswith('conv_'):
        conv_layers_dict = {
            'conv_0_bn_relu/conv_/w': 'block.0.convolutional_block.1.weight',  # first conv layer
            'conv_0_bn_relu/bn_/gamma': 'block.0.convolutional_block.2.weight',  
            'conv_0_bn_relu/bn_/beta': 'block.0.convolutional_block.2.bias',
            'conv_0_bn_relu/bn_/moving_mean': 'block.0.convolutional_block.2.running_mean',
            'conv_0_bn_relu/bn_/moving_variance': 'block.0.convolutional_block.2.running_var',

            'conv_1_bn_relu/conv_/w': 'block.4.convolutional_block.0.weight',  # layer with dropout
            'conv_1_bn_relu/bn_/gamma': 'block.4.convolutional_block.1.weight',  
            'conv_1_bn_relu/bn_/beta': 'block.4.convolutional_block.1.bias',
            'conv_1_bn_relu/bn_/moving_mean': 'block.4.convolutional_block.1.running_mean',
            'conv_1_bn_relu/bn_/moving_variance': 'block.4.convolutional_block.1.running_var',

            'conv_2_bn/conv_/w': 'block.6.convolutional_block.0.weight',  # layer with dropout
            'conv_2_bn/bn_/gamma': 'block.6.convolutional_block.1.weight',  
            'conv_2_bn/bn_/beta': 'block.6.convolutional_block.1.bias',
            'conv_2_bn/bn_/moving_mean': 'block.6.convolutional_block.1.running_mean',
            'conv_2_bn/bn_/moving_variance': 'block.6.convolutional_block.1.running_var',
        }
        name_pt = conv_layers_dict[name_tf]
    return name_pt
    
    
def tf2pt(name_tf, tensor_tf):
    name_pt = tf2pt_name(name_tf)
    num_dimensions = tensor_tf.dim()
    if num_dimensions == 1:
        tensor_pt = tensor_tf
    elif num_dimensions == 5:
        tensor_pt = tensor_tf.permute(4, 3, 0, 1, 2)
    return name_pt, tensor_pt