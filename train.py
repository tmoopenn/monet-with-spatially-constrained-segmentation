import numpy as np
import torch.nn.functional as F
from torch.utils.data import RandomSampler, BatchSampler
import time
from train_options import TrainOptions
from models import create_model
from episodes import get_episodes
#from util.visualizer import Visualizer

def generate_batch(episodes, batch_size):
    total_steps = sum([len(e) for e in episodes])
    print('Total Steps: {}'.format(total_steps))
    # Episode sampler
    # Sample `num_samples` episodes then batchify them with `self.batch_size` episodes per batch
    sampler = BatchSampler(RandomSampler(range(len(episodes)), replacement=True, num_samples=total_steps),
                            batch_size, drop_last=True)
    for indices in sampler:
        episodes_batch = [episodes[x] for x in indices]
        x_t, x_tprev, x_that, ts, thats = [], [], [], [], []
        for episode in episodes_batch:
            # Get one sample from this episode
            t, t_hat = 0, 0
            t, t_hat = np.random.randint(0, len(episode)), np.random.randint(0, len(episode))
            frame = episode[t]
            ### EXAMPLE RESIZING ### 
            #resized_frame = F.interpolate(frame.unsqueeze(0) / 255.0, size=160, mode='bicubic').squeeze(0)
            x_t.append(resized_frame)
        yield torch.stack(x_t).float().to(device) 

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    #dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    #visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        if opt.dynamic_datagen:
            episodes = get_episodes(opt.game, opt.epoch_steps)
            x_batch = generate_batch(episodes, opt.batch_size)
        dataset = x_batch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                #visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                #visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                #if opt.display_id > 0:
                    #visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.