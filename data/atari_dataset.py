import os
import torchvision.transforms.functional as TF
import numpy as np
from torch.utils.data import RandomSampler, BatchSampler
import torch
from utils import util

from data.image_folder import make_dataset
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
                            crop_size=160, # crop is done first
                            load_size=64,  # before resize
                            num_slots=7, display_ncols=7)
        parser.add_argument('--collect_mode', type=str, default='random_agent', help='Specifies whether agent in atari should be random or pretrained agent (pretrained_ppo)')
        parser.add_argument('--game', type=str, default='SpaceInvadersNoFrameskip-v4', help='Atari game to gather frames from')
        parser.add_argument('--dynamic_datagen', action='store_true', help='flag indicating whether train batches will be generated dynamically')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.use_pednet = opt.use_pednet
        self.dynamic_datagen = opt.dynamic_datagen
        self.A_paths = None
        if opt.dynamic_datagen:
            if opt.use_pednet:
                self.A_imgs = self.generate_epoch_episodes_multi_frame()
            else:
                self.A_imgs = self.generate_epoch_episodes()
        else:
            BaseDataset.__init__(self, opt)
            p = os.path.join(opt.dataroot, 'images', 'train' if opt.isTrain else 'test')
            self.A_paths = sorted(make_dataset(p, opt.max_dataset_size))
            if opt.use_pednet:
                self.A_paths = [self.A_paths[i:i+3] for i in range(len(self.A_paths)-3)]

    def _transform(self, img):
        '''
        Applies transforms to input per transforms specified by options 
        Parameters:
            img (PIL image) -- image to apply transforms to
        '''
        # crop at top left corner (0,0) for crop box size of (self.opt.crop_size, self.opt.crop_size) then resize cropped image to self.opt.load_size 
        # expects image in PIL format 
        img = TF.resized_crop(img,  0, 0, self.opt.crop_size, self.opt.crop_size, self.opt.load_size, Image.NEAREST)
        img = TF.to_tensor(img)

        # normalize with mean of 0 and st.dev 1
        img = TF.normalize(img, [0.0] * self.opt.input_nc, [1.0] * self.opt.input_nc)
        return img

    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        path = ""
        if self.use_pednet:
            if not self.dynamic_datagen:
                A_path = self.A_paths[index]
                path = A_path
                A_img_prev, A_img_t, A_img_next = Image.open(A_path[0]).convert('RGB'), Image.open(A_path[1]).convert('RGB'), \
                                Image.open(A_path[2]).convert('RGB')
            else:
                A_img = self.A_imgs[index]
                A_img_prev, A_img_t, A_img_next = A_img[:3,:,:], A_img[3:6,:,:], A_img[6:,:,:]
                A_img_prev, A_img_t, A_img_next = TF.to_pil_image(A_img_prev), TF.to_pil_image(A_img_t), TF.to_pil_image(A_img_next)
            A_prev, A_t, A_next = self._transform(A_img_prev), self._transform(A_img_t), self._transform(A_img_next)
            A = torch.cat((A_prev, A_t, A_next),0)
            return {'A':A, 'A_paths':path}
        else:
            if not self.dynamic_datagen:
                A_path = self.A_paths[index]
                path = A_path
                A_img = Image.open(A_path).convert('RGB')
            else:
                A_img = self.A_imgs[index]
                A_img = TF.to_pil_image(A_img)
            A = self._transform(A_img)
            path = "" if not self.A_paths else self.A_paths
            return {'A': A, 'A_paths': path} 
      

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.dynamic_datagen:
            return len(self.A_imgs)
        return len(self.A_paths)
    
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

    def generate_epoch_episodes_multi_frame(self):
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
            for i in range(0,len(episode)-3,3):
                x_prev, xt, x_next = episode[i:i+3]
                x = torch.cat((x_prev, xt, x_next),0)
                all_frames.append(x)
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
