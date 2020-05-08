"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import torch.nn.functional as F


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)
    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def downsample(frames):
    '''
    Built specifically for downsampling atari frames to 128 x 128 
    Parameters:
        frame (tensor) -- tensor of shape n x c x h x w
    '''
    #frames_tensor = torch.stack(frames)
    downsampled = F.interpolate(frames / 255.0, size=64, mode='nearest')
    return downsampled * 255.0

def visualize_frame(frame):
    '''
    Assume frame is normalized and a torch tensor of shape (c, h, w)
    '''
    import matplotlib.pyplot as plt 
    f = frame.permute(1,2,0).numpy()
    plt.imshow(f)
    plt.show(f)

def extract_patches(x, kernal_size, padding='SAME',sx=1,sy=1) :
    if padding == 'SAME':
        x = F.pad(x, (1,1,1,1))
    elif padding == 'VALID':
        x = x
    else:
        raise ValueError(padding + " not recognized")
    kh, kw = kernal_size
    dh, dw = sy, sx 

    # get all image windows of size (kh, kw) and stride (dh, dw)
    patches = x.unfold(2, kh, dh).unfold(3, kw, dw)
    #print(patches.shape)  # [b, c, h, w, kh, kw]
    # Permute so that channels are next to patch dimension
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [b, h, w, c, kh, kw]
    # View/Reshape as [batch_size, height, width, channels*kh*kw]
    patches = patches.view(*patches.size()[:3], -1)
    return patches