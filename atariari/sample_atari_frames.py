import gym
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from atariari.benchmark.wrapper import AtariARIWrapper
from atariari.benchmark.episodes import get_ppo_rollouts
import argparse
import os
import png
import numpy as np
from skimage.io import imsave
import pdb
import tensorflow as tf
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--ep', default='100', help='Number of atari episode simultations to run')
parser.add_argument('--game', default='PitfallNoFrameskip-v4', help='game to generate samples from')
parser.add_argument('--images', default=os.getcwd() + '/images', help='directory to write images to')

'''
Images: array of 2d numpy array where each 2d numpy array represents a sampled image from the game.
Images are written to the command line argument images directory
'''
def write_images(images, destination_directory):
    ext = '.png'
    if not os.path.isdir(destination_directory):
        try:
            os.mkdir(destination_directory)
        except OSError:
            print("Creation of directory %s failed" % path)
            raise ValueError("Can't Create Directory")

    for i,image in enumerate(images):
        #png.Writer(destination_directory + '/image' + str(i), image)
        imsave(destination_directory + '/image' + str(i) + ext, image)

def sample_images(game, epochs):
        for ep in range(epochs):
            ep_steps, ep_reward = 0, 0
            ep_images = []
            env = AtariARIWrapper(gym.make(game))
            obs = env.reset()
            action = 0
            done = False
            while not done:
                #env.render()
                obs, reward, done, info = env.step(action)
                ep_steps += 1
                ep_reward += reward
                ep_images.append(obs)

                ## GENERATE RANDOM ACTION ##
                #print(env.action_space.n)
                #print(obs.shape)
                action = env.action_space.sample()

                ### SHOWING IMAGE ###
                # print(info)
                #plt.imshow(obs)
                #plt.show(block=False)
                #plt.pause(3)
                #plt.close()
            write_images(ep_images, args.images)

def get_episodes(game, steps=1000):
    eps, eps_labels = get_ppo_rollouts(game, steps)
    return eps, eps_labels


if __name__=="__main__":
    args = parser.parse_args()
    epochs = int(args.ep)
    game = args.game
    eps, eps_labels = get_episodes(game, steps=1)
    print(eps[0][0].shape)

    ### VISUALIZE A SINGLE EPISODE OF PPO AGENT ###
    # ep = eps[0]
    # for frame in ep:
    #     frame = frame.squeeze(0).numpy()
    #     plt.imshow(frame)
    #     plt.show(block=False)
    #     plt.pause(0.01)
    #     plt.close()

    frame = eps[0][0].squeeze(0).numpy()
    print(eps[0][0].numpy().shape)
    plt.imshow(frame)
    plt.show()
    image_tensor = eps[0][0].unsqueeze(0)
    print(image_tensor.shape)
    #cropped = tf.image.crop_to_bounding_box(np.expand_dims(image, axis=len(image.shape)-1), 34, 0, 160, 160)
    resized = F.interpolate(image_tensor / 255.0, size=160, mode='bicubic')
    print(resized.shape, resized.dtype)
    print(resized)
    plt.imshow(resized.squeeze(0).squeeze(0).numpy() * 255)
    plt.show()
