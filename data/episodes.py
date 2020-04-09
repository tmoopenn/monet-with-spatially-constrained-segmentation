'''
Code excerpts borrowed from Atari benchmark developed in paper
Unsupervised State Representation Learning in Atari (https://arxiv.org/abs/1906.08226)
'''

import torch.nn.functional as F
import torch
import numpy as np
import util
import pdb
from atariari.benchmark.episodes import get_ppo_rollouts, get_random_agent_rollouts 


def get_episodes(env_name,
                 steps,
                 seed=42,
                 num_processes=1,
                 num_frame_stack=1,
                 downsample=False,
                 color=True,
                 entropy_threshold=0.6,
                 collect_mode="random_agent",
                 checkpoint_index=-1,
                 min_episode_length=64):

    if collect_mode == "random_agent":
        # List of episodes. Each episode is a list of 160x210 observations
        episodes, episode_labels = get_random_agent_rollouts(env_name=env_name,
                                                             steps=steps,
                                                             seed=seed,
                                                             num_processes=num_processes,
                                                             num_frame_stack=num_frame_stack,
                                                             downsample=False, color=color)

    elif collect_mode == "pretrained_ppo":
        # List of episodes. Each episode is a list of 160x210 observations
        episodes, episode_labels = get_ppo_rollouts(env_name=env_name,
                                                   steps=steps,
                                                   seed=seed,
                                                   num_processes=num_processes,
                                                   num_frame_stack=num_frame_stack,
                                                   downsample=False,
                                                   color=color,
                                                   checkpoint_index=checkpoint_index)


    else:
      assert False, "Collect mode {} not recognized".format(collect_mode)

    # Get indices for episodes that have min_episode_length
    ep_inds = [i for i in range(len(episodes)) if len(episodes[i]) > min_episode_length]
    episodes = [episodes[i] for i in ep_inds]
    
    # Shuffle
    inds = np.arange(len(episodes))
    rng = np.random.RandomState(seed=seed)
    rng.shuffle(inds)

    episodes = [torch.stack(episode) for episode in episodes]
    if downsample:
          episodes = [util.downsample(episode) for episode in episodes]

    #episodes = episodes[inds]
    #pdb.set_trace()
    print("Number of Episodes:", len(episodes))
    print("Frame shape:", episodes[0][0].shape)
    return episodes


