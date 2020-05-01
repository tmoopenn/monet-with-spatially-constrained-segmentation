'''
N. Zhang, et al. A Spatially Constrained Deep Convolutional Neural Network for Nerve Fiber Segmentation 
in Corneal Confocal Microscopic Images Using Inaccurate Annotations, International Symposium on Biomedical Imaging, 2020. In press.
'''
import time 
import numpy as np 
import torch 
import torch.nn as N
import torch.nn.functional as F
from utils.util import extract_patches

def unsupervised_spatial_constraint_loss(x, mask_logits, kernel_size=3, sigma=0.5, positive_threshold=0.51):
    '''
    Parameters:
    x: tensor representing input image of shape (b, c, h, w). Will be converted to grayscale taking max channel 
    mask_logits: logits outputted from segmentation network of shape (b,1,h,w). 
    kernel_size: filter size to use for extracting image patches 
    sigma: standard deviation
    positive_threshold: threshold above which to consider logits as members of the positive class
    '''
    ndim = len(mask_logits.shape) 
    assert ndim in [4], 'only allow 2d images without RGB channel.' # 4dim case [batch_size, c, h, w]
    if type(kernel_size) is int:
        kernel_size = [kernel_size,] * (ndim-2) 
    elif type(kernel_size) is list:
        kernel_size = kernel_size 

    # convert logits to probabilities 
    probs = torch.sigmoid(mask_logits) # (b, c, h, w)
    negative_class = torch.ones(probs.shape) - probs
    probs = torch.cat((probs, negative_class), 1)
    #print(probs[0,:,0,0].sum()) # check value within [0,1]
    #print(probs[0,:,:,:].sum()) # analyze prob sum across image

    # assign labels to probabilities based on positive_threshold 
    #preds = probs > positive_threshold # (b, 1, h, w) In multi-class case, need to take argmax of channel dim
    confs = torch.max(probs,1)[0].unsqueeze(1)
    preds = torch.argmax(probs,1).unsqueeze(1)
    #print("preds",preds.shape)

    p_zmask = extract_patches(torch.ones(confs.shape), kernel_size, padding='SAME')
    p_confs = extract_patches(confs, kernel_size, padding='SAME')
    p_orgs = extract_patches(torch.max(x,dim=1)[0].unsqueeze(1), kernel_size, padding='SAME')
    p_preds = extract_patches(preds, kernel_size, padding='SAME')

    confs = confs.permute(0,2,3,1)
    preds = preds.permute(0,2,3,1)
    x_max = torch.max(x,dim=1)[0].unsqueeze(1).permute(0,2,3,1)

    p_exp = torch.exp(-( x_max - p_orgs)**2 / (2*sigma**2)) # (b, h, w, kh * kw)
    p_exp = p_zmask * p_exp # mask with ones  p_zmask (b, h, w, kh * kw)
    p_mask = 2 * (preds == p_preds).float() - 1 # for labels that match center pixel retain 1 else becomes -1 (b, h, w, kh * kw)

    #print("p_exp", p_exp.shape)
    #print("p_zmask", p_zmask.shape)
    #print("p_mask", p_mask.shape)

    u_ij = p_exp * p_mask # make p_exp entries positive or negative depending on whether they match center pixel (b, h, w, kh * kw)
    #print("u_ij", u_ij.shape)
    P_ij = confs * p_confs # multiply center pixel i by other j kernel pixels (b, h, w, kh * kw)
    #print("P_ij", P_ij.shape)
    F_ij = u_ij * P_ij # u_ij multiplied by confidence scores as per paper  (b, h, w, kh * kw)
    F_ij = torch.sum(F_ij, dim=-1)
    #print("F_ij, confs", F_ij.shape, confs.shape)
    p_exp = torch.sum(p_exp, dim=-1)
    
    # Here F_ij is the sum of center probability and neigbor pixel patch probability (in this case 9 surrounding pixels)
    # this sum contains the term Pi * Pj where i==j however we only want constraint on nearby pixels not center pixel
    # Subtract the term Pi**2 which is same as confs**2
    # In denominator, subtract weight in mu where i == j exp(li - li) = exp(0) = 1
    # Add 1e-9 so no divide by zero 
    F_i = (F_ij - (confs**2).squeeze(-1)) /  (p_exp - 1 + 1e-9) # (b, h, w, 1)
    #print("F_i",F_i.shape)
    sc_loss_map = 1 - F_i 

    return sc_loss_map

