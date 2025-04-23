import torch
import torch.nn as nn
import numpy as np
from .downsampler import Downsampler

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)
    
torch.nn.Module.add = add_module

class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]        

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs: 
                diff2 = (inp.size(2) - target_shape2) // 2 
                diff3 = (inp.size(3) - target_shape3) // 2 
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class GenNoise(nn.Module):
    def __init__(self, dim2):
        super(GenNoise, self).__init__()
        self.dim2 = dim2

    def forward(self, input):
        a = list(input.size())
        a[1] = self.dim2
        # print (input.data.type())

        b = torch.zeros(a).type_as(input.data)
        b.normal_()

        x = torch.autograd.Variable(b)

        return x


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)

class MyReLU(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input_tensor):
        if self.inplace:
            return torch.clamp_(input_tensor, min=0)
        else:
            return torch.clamp(input_tensor, min=0)

class Modulus(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_tensor):
        return torch.abs(input_tensor)

class ArchitecturalActivation(nn.Module):
    """
    Custom activation optimized for architectural features
    Preserves sharp edges and straight lines
    """

    def __init__(self, alpha=0.2, beta=0.5):
        super(ArchitecturalActivation, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.leaky = nn.LeakyReLU(alpha, inplace=False)

    def forward(self, x):
        # Calculate structure tensor to detect edge directions
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

        # Pad to maintain dimensions
        dx = nn.functional.pad(dx, (0, 1, 0, 0), mode='replicate')
        dy = nn.functional.pad(dy, (0, 0, 0, 1), mode='replicate')

        # Edge strength
        edge_strength = torch.sqrt(dx.pow(2) + dy.pow(2))

        # Create adaptive weight
        weight = torch.sigmoid(self.beta * edge_strength)

        # Apply weighted activation - preserves more of the original signal at edges
        return x * weight + self.leaky(x) * (1 - weight)


class FrequencyAwareActivation(nn.Module):
    """
    Treats high and low frequency components differently
    Better for preserving both structural elements and textures
    """

    def __init__(self):
        super(FrequencyAwareActivation, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Simple high-pass filter (Laplacian approximation)
        avg_pool = nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq = torch.abs(x - avg_pool)

        # Apply different activations based on frequency content
        # Gentler on high-frequency components, stronger on low-frequency
        weight = self.sigmoid(high_freq)
        return x * weight + self.relu(x) * (1 - weight)


def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        elif act_fun == 'Modulus':
             return Modulus()
        elif act_fun == 'Architectural':
            return ArchitecturalActivation()
        elif act_fun == 'FrequencyAware':
            return FrequencyAwareActivation()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)