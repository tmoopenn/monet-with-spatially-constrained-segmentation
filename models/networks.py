'''
VAE and U-Net implementations and helper functions borrowed from github repo below
https://github.com/baudm/MONet-pytorch/blob/master/models/networks.py

(VAE) Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. (https://arxiv.org/abs/1312.6114)
(U-Net) Ronneberger et al., "U-net: Convolutional networks for biomedical image segmentation." (https://arxiv.org/abs/1505.04597)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

#### SPATIAL TRANSFORMER ###
def stn_generator(image, z_where, out_dims, inverse=False, box_attn_window_color=None):
    """
    Parameters
    image: input to be transformed 
    z_where: transformation matrix of shape (scale, tx, ty) rotation and translation
    out_dims: size of output grid on which to generate samples from input
    inverse: flag indicating whether transformation matrix z_where should be inverted 
    box_attn_window_color: color of bounding box. Set to None by default so bounding box is not visualized 
    """
    """ spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """

    if box_attn_window_color is not None:
        # draw a box around the attention window by overwriting the boundary pixels in the given color channel
        with torch.no_grad():
            box = torch.zeros_like(image.expand(-1,3,-1,-1))
            c = box_attn_window_color % 3  # write the color bbox in channel c, as model time steps 
            box[:,c,:,0] = 1
            box[:,c,:,-1] = 1
            box[:,c,0,:] = 1
            box[:,c,-1,:] = 1
            # add box to image and clap at 1 if overlap
            image = torch.clamp(image + box, 0, 1)

    # 1. construct 2x3 affine matrix for each datapoint in the minibatch
    theta = torch.zeros(2,3).repeat(image.shape[0], 1, 1).to(image.device)
    # set scaling
    theta[:, 0, 0] = theta[:, 1, 1] = z_where[:,0] if not inverse else 1 / (z_where[:,0] + 1e-9)
    # set translation
    theta[:, :, -1] = z_where[:, 1:] if not inverse else - z_where[:,1:] / (z_where[:,0].view(-1,1) + 1e-9)
    # 2. construct sampling grid
    grid = F.affine_grid(theta, torch.Size(out_dims))
    # 3. sample image from grid using bilinear interpolation by default 
    return F.grid_sample(image, grid)

###############################################################################
# Networks
###############################################################################
class Flatten(nn.Module):
    
    def forward(self, x):
        return x.flatten(start_dim=1)

class ComponentVAE(nn.Module):

    def __init__(self, input_nc, z_dim=16, full_res=False):
        super().__init__()
        self._input_nc = input_nc
        self._z_dim = z_dim
        # full_res = False # full res: 128x128, low res: 64x64
        h_dim = 4096 if full_res else 1024
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc + 1, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(h_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 32)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim + 2, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, input_nc + 1, 1),
        )
        self._bg_logvar = 2 * torch.tensor(0.09).log()
        self._fg_logvar = 2 * torch.tensor(0.11).log()

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    @staticmethod
    def spatial_broadcast(z, h, w):
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)
        # Coordinate axes:
        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)
        # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, x, log_m_k, background=False):
        """
        :param x: Input image
        :param log_m_k: Attention mask logits
        :return: x_k and reconstructed mask logits
        """
        params = self.encoder(torch.cat((x, log_m_k), dim=1))
        z_mu = params[:, :self._z_dim]
        z_logvar = params[:, self._z_dim:]
        z = self.reparameterize(z_mu, z_logvar)

        # "The height and width of the input to this CNN were both 8 larger than the target output (i.e. image) size
        #  to arrive at the target size (i.e. accommodating for the lack of padding)."
        h, w = x.shape[-2:]
        z_sb = self.spatial_broadcast(z, h + 8, w + 8)

        output = self.decoder(z_sb)
        x_mu = output[:, :self._input_nc]
        x_logvar = self._bg_logvar if background else self._fg_logvar
        m_logits = output[:, self._input_nc:]

        return m_logits, x_mu, x_logvar, z_mu, z_logvar


class AttentionBlock(nn.Module):

    def __init__(self, input_nc, output_nc, resize=True):
        super().__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(output_nc, affine=True)
        self._resize = resize

    def forward(self, *inputs):
        downsampling = len(inputs) == 1
        x = inputs[0] if downsampling else torch.cat(inputs, dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = skip = F.relu(x)
        if self._resize:
            x = F.interpolate(skip, scale_factor=0.5 if downsampling else 2., mode='nearest')
        return (x, skip) if downsampling else x


class Attention(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, ngf=64, full_res=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Attention, self).__init__()
        self.downblock1 = AttentionBlock(input_nc + 1, ngf)
        self.downblock2 = AttentionBlock(ngf, ngf * 2)
        self.downblock3 = AttentionBlock(ngf * 2, ngf * 4)
        self.downblock4 = AttentionBlock(ngf * 4, ngf * 8)
        self.full_res = full_res
        if full_res:
            self.downblock5 = AttentionBlock(ngf * 8, ngf * 8)
            # no resizing occurs in the last block of each path
            self.downblock6 = AttentionBlock(ngf * 8, ngf * 8, resize=False)
        else:
            self.downblock5 = AttentionBlock(ngf * 8, ngf * 8, resize=False)

        self.mlp = nn.Sequential(
            nn.Linear(4 * 4 * ngf * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4 * 4 * ngf * 8),
            nn.ReLU(),
        )

        if full_res:
            self.upblock1 = AttentionBlock(2 * ngf * 8, ngf * 8)
        self.upblock2 = AttentionBlock(2 * ngf * 8, ngf * 8)
        self.upblock3 = AttentionBlock(2 * ngf * 8, ngf * 4)
        self.upblock4 = AttentionBlock(2 * ngf * 4, ngf * 2)
        self.upblock5 = AttentionBlock(2 * ngf * 2, ngf)
        # no resizing occurs in the last block of each path
        self.upblock6 = AttentionBlock(2 * ngf, ngf, resize=False)

        self.output = nn.Conv2d(ngf, output_nc, 1)

    def forward(self, x, log_s_k):
        # Downsampling blocks
        x, skip1 = self.downblock1(torch.cat((x, log_s_k), dim=1))
        x, skip2 = self.downblock2(x)
        x, skip3 = self.downblock3(x)
        x, skip4 = self.downblock4(x)
        x, skip5 = self.downblock5(x)
        skip6 = skip5
        # The input to the MLP is the last skip tensor collected from the downsampling path (after flattening)
        if self.full_res:
            _, skip6 = self.downblock6(x)
        # Flatten
        x = skip6.flatten(start_dim=1)
        x = self.mlp(x)
        # Reshape to match shape of last skip tensor
        x = x.view(skip6.shape)
        # Upsampling blocks
        if self.full_res:
            x = self.upblock1(x, skip6)
        x = self.upblock2(x, skip5)
        x = self.upblock3(x, skip4)
        x = self.upblock4(x, skip3)
        x = self.upblock5(x, skip2)
        x = self.upblock6(x, skip1)
        # Output layer
        x = self.output(x)
        x = F.logsigmoid(x)
        return x

class PedNetBaseConvBlock(nn.Module):

    def __init__(self, input_nc, output_nc):
        super(PedNetBaseConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(output_nc, affine=True)
    
    def forward(self, input):
        x = self.conv(input)
        x = self.norm(x)
        return x 

class PedNetEarlyConvBlock(nn.Module):
    
    def __init__(self, input_nc, output_nc):
        super(PedNetEarlyConvBlock, self).__init__()
        self.ped1 = PedNetBaseConvBlock(input_nc, output_nc)
        self.ped2 = PedNetBaseConvBlock(output_nc, output_nc)
    
    def forward(self, input):
        x = self.ped1(input)
        x = self.ped2(x)
        return x 

class PedNet(nn.Module):
    '''
    Ullah, Mohib et al. "Pednet: A spatio-temporal deep convolutional neural network for pedestrian segmentation."
    '''

    def __init__(self, input_nc, output_nc, ngf=32, use_scope=True, prev_timestep_nc=8,
        curr_timestep_nc=32, next_timestep_nc=8):
        """Construct a PedNet Segmentation Network
        Parameters:
            input_nc (int)         -- the number of channels in input images
            output_nc (int)        -- the number of channels in output images
            num_downs (int)        -- the number of downsamplings in PedNet. For example, # if |num_downs| == 7,
                                    image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)              -- the number of filters in the last conv layer
            use_scope (bool)       -- flag indicating whether additional scope channel will be added to input 
            prev_timestep_nc (int) -- the number of channels in conv block for frame at timestep t-1 
            curr_timestep_nc (int) -- the number of channels in conv block for frame at timestep t 
            next_timestep_nc (int) -- the number of channels in conv block for frame at timestep t+1 
           
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(PedNet, self).__init__()
        self.use_scope = use_scope
        
        self.pedblock_prev = PedNetEarlyConvBlock(input_nc, prev_timestep_nc)
        self.pedblock_curr = PedNetEarlyConvBlock(input_nc, curr_timestep_nc)
        self.pedblock_next = PedNetEarlyConvBlock(input_nc, next_timestep_nc)
        #self.pedblock_final_conv = PedNetBaseConvBlock(prev_timestep_nc + curr_timestep_nc + next_timestep_nc, input_down_nc)
        input_down_nc = prev_timestep_nc + curr_timestep_nc + next_timestep_nc

        if use_scope:
            self.downblock1 = AttentionBlock(input_down_nc + 1, ngf)
        else:
            self.downblock1 = AttentionBlock(input_down_nc, ngf)
        self.downblock2 = AttentionBlock(ngf, ngf * 2)
        self.downblock3 = AttentionBlock(ngf * 2, ngf * 4)
        self.downblock4 = AttentionBlock(ngf * 4, ngf * 8)
        self.downblock5 = AttentionBlock(ngf * 8, ngf * 16, resize=True)
        # no resizing occurs in the last block of each path
        self.downblock6 = AttentionBlock(ngf * 16, ngf * 32, resize=False)

        self.mlp = nn.Sequential(
            nn.Linear(4 * 4 * ngf * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4 * 4 * ngf * 32),
            nn.ReLU(),
        )

        self.upblock1 = AttentionBlock(2 * ngf * 32, ngf * 16)
        self.upblock2 = AttentionBlock(2 * ngf * 16, ngf * 8)
        self.upblock3 = AttentionBlock(2 * ngf * 8, ngf * 4)
        self.upblock4 = AttentionBlock(2 * ngf * 4, ngf * 2)
        self.upblock5 = AttentionBlock(2 * ngf * 2, ngf)
        # no resizing occurs in the last block of each path
        self.upblock6 = AttentionBlock(2 * ngf, ngf, resize=False)

        self.output = nn.Conv2d(ngf, output_nc, 1)
    
    def forward(self, xt_prev, xt, xt_next, log_s_k):
        # PedNet Early Blocks 
        x_prev = self.pedblock_prev(xt_prev) # x[:,:3,:,:]
        x_curr = self.pedblock_curr(xt) # x[:,3:6,:,:]
        x_next = self.pedblock_next(xt_next) # x[:,6:,:,:]

        # Concatenate feature maps from all three frames 
        x = torch.cat((x_prev, x_curr, x_next), dim=1)

        # Compress concatenated feature maps from input frames 
        #x = self.pedblock_final_conv(x)

        # Downsampling blocks
        if self.use_scope:
            x, skip1 = self.downblock1(torch.cat((x, log_s_k), dim=1))
        else:
            x, skip1 = self.downblock1(x)
        x, skip2 = self.downblock2(x)
        x, skip3 = self.downblock3(x)
        x, skip4 = self.downblock4(x)
        x, skip5 = self.downblock5(x)
        skip6 = skip5
        # The input to the MLP is the last skip tensor collected from the downsampling path (after flattening)
        _, skip6 = self.downblock6(x)
        # Flatten
        x = skip6.flatten(start_dim=1)
        x = self.mlp(x)
        # Reshape to match shape of last skip tensor
        x = x.view(skip6.shape)
        # Upsampling blocks
        x = self.upblock1(x, skip6)
        x = self.upblock2(x, skip5)
        x = self.upblock3(x, skip4)
        x = self.upblock4(x, skip3)
        x = self.upblock5(x, skip2)
        x = self.upblock6(x, skip1)
        # Output layer
        x = self.output(x)
        x = F.logsigmoid(x)
        return x

class MaskEncoder(nn.Module):

    def __init__(self, input_nc, z_dim=256, ngf=16):
        '''
        Encoder Network consisting of convolutional layers followed by MLP that encodes a mask map.
        Parameters:
        '''
        super(MaskEncoder,self).__init__()
        self.downblock1 = AttentionBlock(input_nc, ngf)
        self.downblock2 = AttentionBlock(ngf, ngf * 2)
        self.downblock3 = AttentionBlock(ngf * 2, ngf * 4)
        self.downblock4 = AttentionBlock(ngf * 4, ngf * 8)
        self.downblock5 = AttentionBlock(ngf * 8, ngf * 8)

        self.mlp = nn.Sequential(
            nn.Linear(4 * 4 * ngf * 8, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim),
        )
    
    def forward(self, x):
        x, _ = self.downblock1(x)
        x, _ = self.downblock2(x)
        x, _ = self.downblock3(x)
        x, _ = self.downblock4(x)
        x, _ = self.downblock5(x) # 4 x 4 x ngf * 8
        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        return x


class LatentWhereMLP(nn.Module):

    def __init__(self, in_z_dim=256, out_where_dim=3):
        '''
        Multi-Layer Perceptron (MLP) for predicting affine transformation matrix. Localisation net equivalent from Spatial Transformer. 
        Parameters:
        '''
        super(LatentWhereMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_where_dim * 2),
        )
    
    def forward(self, x):
        x = self.mlp(x)
        return x

class Self_Attn_MultiC(nn.Module):
    """ Self attention Layer with 1 x 1 Convolutions.
    https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    """
    def __init__(self,in_dim,activation):
        super(Self_Attn_MultiC,self).__init__()
        self.channel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim, kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        '''
        The dimension combinations may seem confusing here if you view query, key and value vectors in the typical language-oriented manner.
        Given a feature maps of shape (c,w,h) we flatten spatial dimensions to achieve a matrix of shape (c, w*h) where each channel
        is stacked vertically on top of each other. With this visualization, we can see that the dimensionality of individual query, key or value vectors 
        in this image domain i.e. each "embedding" is a column s.t. it runs vertically along the channels, contrary to typical shaping where each "embedding" 
        is a row. Thus for a given column, each entry relates to the same spatial position (i,j) but across different channels. 
        '''
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B x (W * H) * C_out
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B x C_out x (W * H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check B x (W*H) x (W*H)
        attention = self.softmax(energy) # B X (W*H) X (W*H) given query_i its attention to all key vectors lies along the row
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C_out X (W*H)

        out = torch.bmm(proj_value,attention.permute(0,2,1) ) # B x C_out x (W*H)
        out = out.view(m_batchsize,C,width,height) # B x C_out x W x H
        
        out = F.logsigmoid(self.gamma*out + x)
        # out = out + x 
        # out = F.log_softmax(out.view(b,c,-1), 2).view_as(x)
        return out,attention

class Self_Attn(nn.Module):
    """ Self attention Layer
    """
    def __init__(self,in_dim, activation, k_dim=64, v_dim=64, a_dim=64):
        super(Self_Attn,self).__init__()
        self.in_dim = in_dim
        self.activation = activation
        self.k_dim = k_dim 
        self.a_dim = a_dim
        self.v_dim = v_dim
        
        self.Q = nn.Linear(in_dim, k_dim)
        self.K = nn.Linear(in_dim, k_dim)
        self.V = nn.Linear(in_dim, in_dim)

        self.softmax  = nn.Softmax(dim=-1) 
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self,x,X):
        """
            inputs :
                x : input image patches( B x W x H x (c*kw*kh) ). 
                    Transformed to image patches of flattened shape(c * kw * kh) for each W x H position in image 
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        b, w, h, n = x.size()
        proj_query = self.Q(x).view(b, -1, self.k_dim) # b x w x h x k_dim then view as b x (w*h) x k_dim 
        proj_key = self.K(x).view(b, -1, self.k_dim)   # b x w x h x k_dim then view as b x (w*h) x k_dim
        energy = torch.bmm(proj_query, proj_key.permute(0,2,1)) # b x (w*h) x (w*h)
        attn = self.softmax(energy) 
        proj_value = self.V(x).view(b, -1, self.v_dim) # b x w x h x n

        out = torch.bmm(attn, proj_value) # b x (w*h) x v_dim
        out = out.view(b, w, h, -1) # b x w x h x v_dim

        # MAYBE CHANGE, Here we are summing scores of neighboring pixels in attn window to get a single scalar value for pixel at pos(w,h) then add to input pixel values.
        out = torch.max(out, dim=-1)[0].unsqueeze(1) # b x 1 w x h
        out = F.logsigmoid(self.gamma*out + X)
        # out = out + x 
        # out = F.log_softmax(out.view(b,c,-1), 2).view_as(x)
        return out, attn


