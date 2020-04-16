import os
import torchvision.transforms.functional as TF
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
import torch
import util 

from data.base_dataset import BaseDataset
from PIL import Image
from .episodes import get_episodes

class AtariDataset(BaseDataset):
    """This dataset class dynamically generates frames from the atari framework and should only be used 
    with the options flag dynamic-datagen set to true. It generates frames for one epoch.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=3, output_nc=3,
                            crop_size=180, # crop is done first
                            load_size=128,  # before resize
                            num_slots=7, display_ncols=7)
        parser.add_argument('--collect_mode', type=str, default='random_agent', help='Specifies whether agent in atari should be random or pretrained agent (pretrained_ppo)')
        parser.add_argument('--game', type=str, default='SpaceInvadersNoFrameskip-v4', help='Atari game to gather frames from')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_imgs = self.generate_epoch_episodes()

    def _transform(self, img):
        '''
        Applies transforms to input per transforms specified by options 
        Parameters:
            img (PIL image) -- image to apply transforms to
        '''
        # crop at top left corner (0,34) for crop box size of (self.opt.crop_size, self.opt.crop_size) then resize cropped image to self.opt.load_size 
        # expects image in PIL format 
        img = TF.resized_crop(img,  20, 0, self.opt.crop_size, self.opt.crop_size, self.opt.load_size, Image.BILINEAR)
        img = TF.to_tensor(img)

        # should we normalize with mean of 0 and st.dev 1??
        #img = TF.normalize(img, [0.5] * self.opt.input_nc, [0.5] * self.opt.input_nc)
        return img

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_img = self.A_imgs[index]
        A_img = TF.to_pil_image(A_img)
        A = self._transform(A_img)
        return {'A': A, 'A_paths': ""}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_imgs)
    
    def generate_epoch_episodes(self):
        '''
        Generates a list of sequential frames across different episodes 
        returns: [tensor (c,h,w)] -> a list of 3D tensors where each tensor represents a frame
        '''
        # [[torch tensors (3, h, w)]]
        episodes = get_episodes(self.opt.game, self.opt.epoch_steps, collect_mode=self.opt.collect_mode)
        total_steps = sum([len(e) for e in episodes])
        print('Total Steps: {}'.format(total_steps))
        all_frames = []
        for episode in episodes:
            for frame in episode:
                all_frames.append(frame)
        return all_frames
    
    def generate_epoch_episodes_in_batches(self):
        '''
        Generates batches of frames in randomized order by sampling frames from multiple episodes
        returns: 5-dimensional tensor (n, b, c, h, w)  where n is number of batches, b is batch size, 
        c is number of channels of frame and h and w are height and width of frame
        '''
        # [[torch tensors (3, h, w)]]
        episodes = get_episodes(self.opt.game, self.opt.epoch_steps)
        total_steps = sum([len(e) for e in episodes])
        print('Total Steps: {}'.format(total_steps))
        # Episode sampler
        # Sample `num_samples` frames then batchify them with `self.batch_size` frames per batch
        sampler = BatchSampler(RandomSampler(range(len(episodes)), replacement=True, num_samples=total_steps),
                                self.opt.batch_size, drop_last=True)
        all_batches = []
        for indices in sampler:
            episodes_batch = [episodes[x] for x in indices]
            x_t, x_tprev, x_that, ts, thats = [], [], [], [], []
            for episode in episodes_batch:
                # Get one sample from this episode
                t, t_hat = 0, 0
                t, t_hat = np.random.randint(0, len(episode)), np.random.randint(0, len(episode))
                frame = episode[t]
                x_t.append(frame)
            x_batch = torch.stack(x_t).float()
            all_batches.append(x_batch)
        return torch.stack(all_batches)
