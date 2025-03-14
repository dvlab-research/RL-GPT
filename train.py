import argparse
#from spinup_utils.mpi_tools import mpi_fork
import gym
import utils
import os
from spinup_utils.run_utils import setup_logger_kwargs
#from minecraft import MinecraftEnv
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic arguments for PPO
    parser.add_argument('--gamma', type=float, default=0.99) # discount
    parser.add_argument('--target-kl', type=float, default=0.5) # kl upper bound for updating policy
    parser.add_argument('--seed', '-s', type=int, default=7) # random seed for both np, torch and env
    parser.add_argument('--cpu', type=int, default=1) # number of workers, should be 1
    parser.add_argument('--gpu', default='0') # -1 if use cpu, otherwise select the gpu id
    parser.add_argument('--steps', type=int, default=1000) # sample steps per PPO epoch (buffer size * workers)
    parser.add_argument('--epochs', type=int, default=1000) # PPO epoch number
    parser.add_argument('--save-path', type=str, default='checkpoint') # save dir for model&data. Use /sharefs/baaiembodied/xxx on server
    parser.add_argument('--exp-name', type=str, default='ppo') # experiment log name
    

    # arguments for tasks
    parser.add_argument('--task', type=str, default='harvest_milk_with_empty_bucket_and_cow') # programmatic task_id, for single task
    parser.add_argument('--horizon', type=int, default=200) # task horizon. It is 500 in the MineCLIP released code
    parser.add_argument('--use-multi-task', type=int, default=0) # run different tasks across workers
    parser.add_argument('--tasks-config-path', type=str, default='multi_env_config/harvest_milk.yaml') # path to load multi-task configs
    

    # CLIP model and agent model config
    parser.add_argument('--clip-config-path', type=str, default='mineclip_official/config.yml')
    parser.add_argument('--clip-model-path', type=str, default='mineclip_official/adjust.pth')
    parser.add_argument('--agent-model', type=str, default='mineagent') # agent architecture: mineagent, cnn
    parser.add_argument('--agent-config-path', type=str, default='mineagent/conf.yaml') # for mineagent
    parser.add_argument('--actor-out-dim', type=int, nargs='+', default=[12,3])
    ''' 
    actor output dimensions. mineagent official: [3,3,4,25,25,8]; my initial implement: [56,3]
    mineagent with clipped camera space: [3,3,4,5,3] or [12,3]
    should modify transform_action() in minecraft.py together with this arg
    '''

    # reward weights
    parser.add_argument('--reward-success', type=float, default=100.)
    parser.add_argument('--reward-clip', type=float, default=1.)
    parser.add_argument('--clip-reward-mode', type=str, default='direct') # how to compute clip reward
    parser.add_argument('--reward-step', type=float, default=-1.) # per-step penalty
    parser.add_argument('--use-dense', type=int, default=0) # use dense reward
    parser.add_argument('--reward-dense', type=float, default=1.) # dense reward weight

    # self-imitation learning
    parser.add_argument('--imitate-buf-size', type=int, default=500) # max num of traj to store
    parser.add_argument('--imitate-batch-size', type=int, default=1000) # batchsize for imitation learning
    parser.add_argument('--imitate-freq', type=int, default=100) # how many ppo epochs to run self-imitation
    parser.add_argument('--imitate-epoch', type=int, default=1) # how many self-imitation epochs
    parser.add_argument('--imitate-success-only', type=int, default=0) # save only success trajs into imitation buffer
    
    # arguments for related research works
    parser.add_argument('--save-all-data', type=int, default=0) # save all the collected experience
    parser.add_argument('--save-expert-data', type=int, default=0) # save experience in self-imitation buffer
    parser.add_argument('--save-raw-rgb', type=int, default=1) # save rgb images when save the above data; save gif for debug
    parser.add_argument('--use-ss-reward', type=int, default=0) # experiment for pretrained SS-transformer
    parser.add_argument('--ss-k', type=int, default=10) # prediction horizon for SS transformer
    parser.add_argument('--ss-model-path', type=str, default=
        'ss_transformer/trained_on_youtube_interval_1_blocksize_10.pth') # pretrained SS model path

    args = parser.parse_args()
    #print(args)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    args.save_path = os.path.join(args.save_path, '{}-{}-seed{}'.format(args.exp_name, args.task, args.seed))
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    pth = os.path.join(args.save_path, 'gif')
    if not os.path.exists(pth):
        os.mkdir(pth)
    pth = os.path.join(args.save_path, 'model')
    if not os.path.exists(pth):
        os.mkdir(pth)
    pth = os.path.join(args.save_path, 'experience_buffer')
    if not os.path.exists(pth):
        os.mkdir(pth)

    #mpi_fork(args.cpu)  # run parallel code with mpi
    args.exp_name = args.exp_name + '_' + args.task
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # set gpu device
    if args.gpu == '-1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))
    print('Using device:', device)

    from ppo_selfimitate_clip import ppo_selfimitate_clip
    print('Training ppo_selfimitate_clip.')
    ppo_selfimitate_clip(args,
        gamma=args.gamma, save_path=args.save_path, target_kl=args.target_kl,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs, device=device, 
        clip_config_path=args.clip_config_path, clip_model_path=args.clip_model_path,
        agent_config_path=args.agent_config_path)