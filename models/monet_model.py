"""C. P. Burgess et al., "MONet: Unsupervised Scene Decomposition and Representation," pp. 1–22, 2019."""
'''
Monet implementation based on implementation from github repo below:
https://github.com/baudm/MONet-pytorch/blob/master/models/monet_model.py
'''
from itertools import chain

import torch
from torch import nn, optim
from utils.util import extract_patches
from utils.spatial_constraint_loss import unsupervised_spatial_constraint_loss

from .base_model import BaseModel
from . import networks


class MONetModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.
        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.set_defaults(batch_size=1, lr=1e-4, display_ncols=7, niter_decay=0,
                            dataset_mode='atari', niter=int(64e6 // 7e4))
        parser.add_argument('--num_slots', metavar='K', type=int, default=7, help='Number of supported slots')
        parser.add_argument('--z_dim', type=int, default=16, help='Dimension of individual z latent per slot')
        parser.add_argument('--epoch_steps', type=int, default=100, help='Total number of steps to collect across episodes')
        parser.add_argument('--full_res', action='store_true', default=False, help='Specifies whether model linear layers should expect image size of 64 (False) or 128 (True)')
        parser.add_argument('--attn_window_size', default=3, help='Size of windows to use for self-attention network.')
        
        if is_train:
            parser.add_argument('--beta', type=float, default=0.5, help='weight for the encoder KLD')
            parser.add_argument('--gamma', type=float, default=0.5, help='weight for the mask KLD')
            parser.add_argument('--alpha', type=float, default=0.2, help='weight for the spatial_constraint penalty')
        return parser

    def __init__(self, opt):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.loss_names = ['E', 'D', 'mask']
        self.use_pednet = opt.use_pednet
        self.attn_window_size = opt.attn_window_size
        if self.use_pednet:
            self.visual_names = ['m{}'.format(i) for i in range(opt.num_slots)] + \
                                ['x{}'.format(i) for i in range(opt.num_slots)] + \
                                ['xm{}'.format(i) for i in range(opt.num_slots)] + \
                                ['x_prev','x_t', 'x_next', 'x_tilde']
        else:
            self.visual_names = ['m{}'.format(i) for i in range(opt.num_slots)] + \
                            ['x{}'.format(i) for i in range(opt.num_slots)] + \
                            ['xm{}'.format(i) for i in range(opt.num_slots)] + \
                            ['x', 'x_tilde']
        self.model_names = ['Attn', 'CVAE']
        if opt.use_pednet:
            self.netAttn = networks.init_net(networks.PedNet(opt.input_nc,1), gpu_ids=self.gpu_ids)
        else:
            self.netAttn = networks.init_net(networks.Attention(opt.input_nc, 1), gpu_ids=self.gpu_ids)
        self.netCVAE = networks.init_net(networks.ComponentVAE(opt.input_nc, opt.z_dim, opt.full_res), gpu_ids=self.gpu_ids)
        self.eps = torch.finfo(torch.float).eps
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        if self.isTrain:  # only defined during training time
            self.criterionKL = nn.KLDivLoss(reduction='batchmean')
            self.criterionSC = unsupervised_spatial_constraint_loss
            self.optimizer = optim.RMSprop(chain(self.netAttn.parameters(), self.netCVAE.parameters()), lr=opt.lr)
            self.optimizers = [self.optimizer]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        if self.use_pednet:
            self.x_prev, self.x_t, self.x_next = input['A'][:,:3,:,:], input['A'][:,3:6,:,:], input['A'][:,6:,:,:]
            self.x_prev = self.x_prev.to(self.device)
            self.x_t = self.x_t.to(self.device)
            self.x_next = self.x_next.to(self.device)
        else:
            self.x = input['A'].to(self.device)
        self.image_paths = input['A_paths']

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.loss_E = 0
        self.x_tilde = 0
        b = []
        m = []
        m_tilde_logits = []

        if self.use_pednet:
            # Initial s_k = 1: shape = (N, 1, H, W)
            shape = list(self.x_t.shape)
            shape[1] = 1
            log_s_k = self.x_t.new_zeros(shape) # log(1) = 0
        else:
            # Initial s_k = 1: shape = (N, 1, H, W)
            shape = list(self.x.shape)
            shape[1] = 1
            log_s_k = self.x.new_zeros(shape) # log(1) = 0

        for k in range(self.opt.num_slots):
            # Derive mask from current scope
            if k != self.opt.num_slots - 1:
                if self.use_pednet:
                    log_alpha_k = self.netAttn(self.x_prev, self.x_t, self.x_next, log_s_k)
                else:
                    log_alpha_k = self.netAttn(self.x, log_s_k)
                #mask_patches = extract_patches(log_alpha_k.exp(), (self.attn_window_size, self.attn_window_size),padding='SAME')
                #log_alpha_k, attention = self.netSelfAttn(mask_patches, log_alpha_k.exp()) # (b,1,w,h) (b,(w*h),(w*h)) 
                #log_alpha_k, attention = self.netSelfAttn(log_alpha_k.exp())
                log_m_k = log_s_k + log_alpha_k

                # Apply self-attention to scoped mask so don't attend to previously explained components
                # extract patches 
                #mask_patches = extract_patches(log_m_k.exp(), (self.attn_window_size, self.attn_window_size),padding='SAME')
                # apply self-attention layer, return mask_pixel_scores which is
                # pixel-wise attention scores added to mask logits and attention tensor
                #mask_pixel_scores, attention = self.netSelfAttn(mask_patches, log_m_k.exp()) # (b,1,w,h) (b,(w*h),(w*h)) 
                #log_m_k = mask_pixel_scores

                # TODO: Use self-attended scope mask to update next scope 
                # Compute next scope
                log_s_k += (1. - log_alpha_k.exp()).clamp(min=self.eps).log()
            else:
                log_m_k = log_s_k

            # Get component and mask reconstruction, as well as the z_k parameters
            if self.use_pednet:
                m_tilde_k_logits, x_mu_k, x_logvar_k, z_mu_k, z_logvar_k = self.netCVAE(self.x_t, log_m_k, k == 0)
            else:
                m_tilde_k_logits, x_mu_k, x_logvar_k, z_mu_k, z_logvar_k = self.netCVAE(self.x, log_m_k, k == 0)
            
            # Solution for minimizing negative ELBO in the case of diagonal multivariate normal (approximation ) and normal distribution (target)
            # ∑ -1/2(1 + log(sigma^2) - mu^2 - sigma^2)
            # KLD is additive for independent distributions
            self.loss_E += -0.5 * (1 + z_logvar_k - z_mu_k.pow(2) - z_logvar_k.exp()).sum()

            m_k = log_m_k.exp()
            x_k_masked = m_k * x_mu_k

            # Exponents for the decoder loss
            if self.use_pednet:
                b_k = log_m_k - 0.5 * x_logvar_k - (self.x_t - x_mu_k).pow(2) / (2 * x_logvar_k.exp())
            else:
                b_k = log_m_k - 0.5 * x_logvar_k - (self.x - x_mu_k).pow(2) / (2 * x_logvar_k.exp())
            b.append(b_k.unsqueeze(1))

            # Get outputs for kth step
            setattr(self, 'm{}'.format(k), m_k * 2. - 1.) # shift mask from [0, 1] to [-1, 1]
            setattr(self, 'x{}'.format(k), x_mu_k)
            setattr(self, 'xm{}'.format(k), x_k_masked)

            # Iteratively reconstruct the output image
            self.x_tilde += x_k_masked
            # Accumulate
            m.append(m_k)
            m_tilde_logits.append(m_tilde_k_logits)

        self.b = torch.cat(b, dim=1)
        self.m = torch.cat(m, dim=1)
        self.m_tilde_logits = torch.cat(m_tilde_logits, dim=1)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        if self.use_pednet:
            n = self.x_t.shape[0]
        else:
            n = self.x.shape[0]
        self.loss_E /= n
        self.loss_D = -torch.logsumexp(self.b, dim=1).sum() / n
        self.loss_mask = self.criterionKL(self.m_tilde_logits.log_softmax(dim=1), self.m)
        #self.loss_SC = 0
        #for i in range(self.m.shape[1]):
            #if self.use_pednet:
                #self.loss_SC += unsupervised_spatial_constraint_loss(self.x_t, self.m[:,i,:,:].unsqueeze(1))
            #else:
                #self.loss_SC += self.criterionSC(self.x, torch.cat((self.m[:,i,:,:].unsqueeze(1), self.m_tilde_logits[:,i,:,:].unsqueeze(1)), dim=1))
                #self.loss_SC += self.criterionSC(self.x, self.m[:,i,:,:].unsqueeze(1))
        #self.loss_SC = torch.sum(self.loss_SC, dim=tuple(range(1,self.loss_SC.ndim))) 
        #self.loss_SC = self.loss_SC.sum() / n
        loss = self.loss_D + self.opt.beta * self.loss_E + self.opt.gamma * self.loss_mask + self.opt.alpha
        loss.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        self.optimizer.zero_grad()   # clear network G's existing gradients
        self.backward()              # calculate gradients for network G
        self.optimizer.step()        # update gradients for network G
