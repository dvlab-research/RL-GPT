import utils
import os
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
#from spinup_utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
#from spinup_utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup_utils.logx import EpochLogger
from PIL import Image
import imageio
#from clip_model import build_model, tokenize_batch
#from torchvision.transforms import Resize 
#from skimage.transform import resize
from mineclip_official import build_pretrain_model, tokenize_batch, torch_normalize
from minecraft import MinecraftEnv, preprocess_obs, transform_action
from mineagent.batch import Batch
from mineagent import features, SimpleFeatureFusion, MineAgent, MultiCategoricalActor, Critic
import copy
import pickle


# PPO buffer
# for mineagent: observation is stored with Batch
# for CNN actor: observation should be processed by torch_normalize before save
class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, act_dim, size=1000, gamma=0.99, lam=0.95, agent_model='mineagent', obs_dim=None):
        self.agent_model = agent_model
        if agent_model == 'mineagent':
            self.obs_buf = [Batch() for i in range(size)]#np.zeros(utils.combined_shape(size, obs_dim), dtype=np.float32)
        else:
            self.obs_buf = np.zeros(utils.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(size, act_dim), dtype=np.int)
        #self.act2_buf = np.zeros(utils.combined_shape(size, act2_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        #self.act2_buf[self.ptr] = act[1]
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def modify_trajectory_rewards(self, rews):
        """
        modify the recently saved rewards with a numpy array: rews.
        should be called after many store(), and before finish_path().
        """
        assert self.ptr - self.path_start_idx == len(rews)
        self.rew_buf[self.path_start_idx: self.ptr] = rews

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = utils.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = utils.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf) #mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std


        if self.agent_model == 'mineagent':
            data = dict(act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
            rtn =  {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
            rtn['obs'] = Batch.cat(self.obs_buf)
            #print(rtn)
        else:
            data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
            rtn = {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
        return rtn


# self-imitation learning buffer
# for CNN actor: observation should be processed by torch_normalize before save
class SelfImitationBuffer:
    def __init__(self, act_dim, size=500, imitate_success_only=True, agent_model='mineagent'):
        '''
        each saved item is a trajectory: act_buf [[len, act_dim], ...]
        '''
        self.obs_buf = []
        self.act_buf = [] # np.zeros(utils.combined_shape(size, act_dim), dtype=np.int)
        self.ret_buf = [] # returns
        self.success_buf = []
        self.cur_size, self.max_size = 0, size
        self.baseline = 0.
        self.success_rate = 0.
        self.avg_return = 0.
        self.imitate_success_only = imitate_success_only

        #self.rgb_buf = []
        self.i_saved_traj = 0
        self.agent_model = agent_model

    # eval the trajectory performance and decide to store
    def eval_and_store(self, obs, act, ret, success, rgb=None, save_dir=None):
        '''
        store if success or episode return >= baseline
        if the buffer is full, first-in-first-out
        '''
        if self.cur_size > 0:
            self.baseline = np.mean(self.ret_buf) + 2*np.std(self.ret_buf)
        if success or ((not self.imitate_success_only) and (ret >= self.baseline)):
            assert self.cur_size <= self.max_size
            self.obs_buf.append(obs)
            self.act_buf.append(act)
            self.ret_buf.append(ret)
            self.success_buf.append(success)
            self.success_rate = np.mean(self.success_buf)
            self.avg_return = np.mean(self.ret_buf)
            #self.rgb_buf.append(rgb)

            if self.cur_size < self.max_size:
                self.cur_size += 1
            else: # FIFO
                del(self.obs_buf[0])
                del(self.act_buf[0])
                del(self.ret_buf[0])
                del(self.success_buf[0])
                #del(self.rgb_buf[0])

            # save the expert trajectory
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                pth = os.path.join(save_dir, 'traj_{}.pth'.format(self.i_saved_traj))
                pickle.dump([obs, act, ret, success, rgb], open(pth, 'wb'))
                self.i_saved_traj += 1


        #print(self.cur_size, len(self.obs_buf), self.baseline, self.success_rate,
        #    obs.shape, act.shape, ret)

    # get all the data for training. 
    # convert the trajectory list [N * [len, dim]] to transition array [N', dim]
    def get(self):
        assert self.cur_size > 0
        act_ = np.concatenate(self.act_buf)
        if self.agent_model == 'mineagent':
            obs_ = Batch.cat(self.obs_buf)
            rtn = {
                'act': torch.as_tensor(act_, dtype=torch.long),
                'obs': obs_
            }
        else:
            obs_ = np.concatenate(self.obs_buf)
            rtn = {
                'act': torch.as_tensor(act_, dtype=torch.long),
                'obs': torch.as_tensor(obs_, dtype=torch.float32)
            }
        
        return rtn


# 10/9 update:
# maintain the text embedding 
# compute images embedding (with 15 empty frames at begin) for a trajectory 
# compute constrastive intrinsic rewards for all the steps with a moving window
class CLIPReward:
    def __init__(self, clip_model, device, text):
        self.clip_model = clip_model
        self.device = device
        self.text = text

        # load negative prompts
        with open('negative_prompts.txt', 'r') as f:
            self.neg_text = f.read().splitlines()
        #print(self.text, self.neg_text)
        
        # create initial 15 empty frames before env reset
        #video = torch_normalize(np.zeros([1, 15, 3, 160, 256])).to(self.device)
        with torch.no_grad():
            #self.imgs_emb = self.clip_model.image_encoder(torch.as_tensor(video, dtype=torch.float))
            # pre-compute text embedding
            self.text_emb = self.clip_model.text_encoder(tokenize_batch(self.text + self.neg_text).to(self.device))
            #print(tokenize_batch(self.text + self.neg_text), self.text_emb)
            assert self.text_emb.shape[0] == 1+len(self.neg_text)

    '''
    # update the images queue when calling env.reset() or step()
    # the encoding for the new image is pre-computed in the env wrapper
    def update_obs(self,emb):
        assert emb.shape[0] == 1 and emb.shape[1] == 1 # (1,1,512)
        self.imgs_emb = torch.cat((self.imgs_emb[:,1:], torch.as_tensor(emb, device=self.device)), 1)
    '''

    # compute all the embeddings for a trajectory, concat the 15 empty frames at beginning
    # input imgs should be preprocessed by torch_normalize
    def compute_all_embeddings(self, imgs):
        #assert imgs.dtype is np.int
        video_begin = torch_normalize(np.zeros([1, 15, 3, 160, 256])).to(self.device) # pad 15 frames before reset
        video = imgs.to(self.device) # (1, N, 3, 160, 256)
        video = torch.cat((video_begin, video), 1) # (1, 15+N, 3, 160, 256)
        #print(video.shape)

        with torch.no_grad():
            self.imgs_emb = self.clip_model.image_encoder(torch.as_tensor(video, dtype=torch.float)) # (1, 15+N, 512)
            #print(self.imgs_emb.shape)


    # compute the intrinsic reward for a 16-frames window
    # mode: direct, direct-naive and delta in minedojo paper
    def reward(self, imgs_emb_window, mode='direct'):
        with torch.no_grad():
            v_emb = self.clip_model.temporal_encoder(imgs_emb_window)
            adapted_video, adapted_text = self.clip_model.reward_adapter(v_emb, self.text_emb)
            v_f = adapted_video / adapted_video.norm(dim=1, keepdim=True) # (1, 512)
            v_f = self.clip_model.logit_scale.exp()*v_f
            t_f= adapted_text / adapted_text.norm(dim=1, keepdim=True) # (1, 512)
            #print(v_f.shape, t_f.shape)
            logits_per_video = v_f @ t_f.t() # (1,32)
            prob = F.softmax(logits_per_video, dim=-1)[0][0].detach().cpu().numpy() # P(video corresponds to the prompt)
            
        if mode=='direct':
            assert self.text_emb.shape[0] == 32
            r_clip = max(prob - 1./32, 0)
        else:
            raise NotImplementedError
        return r_clip

    # compute reward sequence for a trajectory, after calling compute_all_embeddings
    def compute_all_rewards(self, mode='direct'):
        r = []
        for i in range(self.imgs_emb.shape[1]-16):
            r.append(self.reward(self.imgs_emb[:, i+1:i+17], mode=mode)) # skip the reset step
        return np.array(r)

    '''
    # reset the queue when env.reset
    def reset(self):
        video = torch_normalize(np.zeros([1, 16, 3, 160, 256])).to(self.device)
        with torch.no_grad():
            self.imgs_emb = self.clip_model.image_encoder(torch.as_tensor(video, dtype=torch.float))
    '''



'''
PPO algorithm implementation:
for every epoch, first play the game to collect trajectories
until the buffer is full, then update the actor and the critic for sevaral steps using the buffer.

ppo_clip uses mineclip intrinsic reward for sparse reward tasks.
'''
def ppo_selfimitate_clip(args, seed=0, device=None, 
        steps_per_epoch=400, epochs=500, gamma=0.99, clip_ratio=0.2, pi_lr=1e-4, vf_lr=1e-4,  
        train_pi_iters=80, train_v_iters=80, lam=0.95, max_ep_len=1000,
        target_kl=0.01, save_freq=5, logger_kwargs=dict(), save_path='checkpoint', 
        clip_config_path='', clip_model_path='', agent_config_path=''):

    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        device: cpu or cuda gpu device for training NN

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """


    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    #setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())


    # Random seed
    #seed += 10000 * proc_id()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


    # load pretrained mineclip model
    clip_config = utils.get_yaml_data(clip_config_path)
    model_clip = build_pretrain_model(
        image_config = clip_config['image_config'],
        text_config = clip_config['text_config'],
        temporal_config = clip_config['temporal_config'],
        adapter_config = clip_config['adaptor_config'],
        state_dict = torch.load(clip_model_path)
    ).to(device)
    model_clip.eval()
    print('MineCLIP model loaded.')


    # Instantiate environment
    env = MinecraftEnv(
        task_id=args.task,
        image_size=(160, 256),
        max_step=args.horizon, 
        clip_model=model_clip if (args.agent_model == 'mineagent') else None, 
        device=device,
        seed=seed,
        dense_reward=bool(args.use_dense)
    )
    obs_dim = env.observation_size
    env_act_dim = env.action_size
    agent_act_dim = len(args.actor_out_dim)
    print('Task prompt:', env.task_prompt)
    #logger.log('env: obs {}, act {}'.format(env.observation_space, env.action_space))


    # Create actor-critic agent
    if args.agent_model == 'mineagent':
        agent_config = utils.get_yaml_data(agent_config_path)
        feature_net_kwargs = agent_config['feature_net_kwargs']
        feature_net = {}
        for k, v in feature_net_kwargs.items():
            v = dict(v)
            cls = v.pop("cls")
            cls = getattr(features, cls)
            feature_net[k] = cls(**v, device=device)
        feature_fusion_kwargs = agent_config['feature_fusion']
        feature_net = SimpleFeatureFusion(
            feature_net, **feature_fusion_kwargs, device=device
        )
        feature_net_v = copy.deepcopy(feature_net) # actor and critic do not share
        actor = MultiCategoricalActor(
            feature_net,
            action_dim=args.actor_out_dim, #[3, 3, 4, 25, 25, 8],
            device=device,
            **agent_config['actor'],
            activation='tanh',
        )
        critic = Critic(
            feature_net_v,
            action_dim=None,
            device=device,
            **agent_config['actor'],
            activation='tanh'
        )
        mine_agent = MineAgent(
            actor=actor, 
            critic=critic,
            deterministic_eval=False
        ).to(device) # use the same stochastic policy in training and test
        mine_agent.eval()
    elif args.agent_model == 'cnn':
        mine_agent = utils.CNNActorCritic(
            action_dim=args.actor_out_dim,
            deterministic_eval=False
        ).to(device)
        mine_agent.eval()
    else:
        raise NotImplementedError

    # Sync params across processes
    #sync_params(ac)

    # Count variables
    var_counts = (#utils.count_vars(actor), utils.count_vars(critic),
        utils.count_vars(mine_agent), utils.count_vars(model_clip))
    #logger.log('\nNumber of parameters: \t actor: %d, \t critic: %d, \t  agent: %d, \t mineclip: %d\n'%var_counts)
    logger.log('\nNumber of parameters: \t agent: %d, \t mineclip: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = steps_per_epoch
    buf = PPOBuffer(agent_act_dim, local_steps_per_epoch, gamma, lam, args.agent_model, obs_dim)

    # set up imitation buffer
    imitation_buf = SelfImitationBuffer(agent_act_dim, args.imitate_buf_size, args.imitate_success_only, args.agent_model)
    

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'].to(device), \
                                data['adv'].to(device), data['logp'].to(device)
        if args.agent_model == 'mineagent':
            obs.to_torch(device=device)
        else:
            obs = obs.to(device)

        # Policy loss
        pi = mine_agent(obs).dist
        logp = pi.log_prob(act)
        #print('logp, logp_old = ', logp, logp_old)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret'].to(device)
        if args.agent_model == 'mineagent':
            obs.to_torch(device=device)
            obs_ = obs.obs
        else:
            obs_ = obs.to(device)
        return ((mine_agent.critic(obs_) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = torch.optim.Adam(mine_agent.actor.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(mine_agent.critic.parameters(), lr=vf_lr)
    #optimizer = torch.optim.Adam(mine_agent.parameters(), lr=lr)


    # a training epoch
    def update():
        mine_agent.train()

        data = buf.get() # dict

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()


        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            #mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = pi_info['kl'] #mpi_avg(pi_info['kl'])
            #logger.log('kl={}'.format(kl))
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl, kl=%f.'%(i, kl))
                break
            loss_pi.backward()
            #mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)


        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))



    # set up function for computing self-imitation loss
    # use the batch indexed by idxs in data
    def compute_loss_imitation(data, idxs):
        obs, act = data['obs'][idxs], data['act'][idxs].to(device)
        if args.agent_model == 'mineagent':
            obs.to_torch(device=device)
        else:
            obs = obs.to(device)
        pi = mine_agent(obs).dist
        loss_imitation = pi.imitation_loss(act)
        return loss_imitation

    # training step for self-imitation learning
    def update_imitation():
        mine_agent.train()
        data = imitation_buf.get()
        n_data = data['act'].shape[0]
        n_iter = max(int(n_data / args.imitate_batch_size), 1) # iterations to train for 1 epoch
        #print('data', data, n_iter, n_data)
        for i in range(n_iter):
            pi_optimizer.zero_grad()
            idxs = np.random.randint(0, n_data, size=args.imitate_batch_size)
            #print('training batch', data['act'][idxs], data['obs'][idxs])
            loss_imitation = compute_loss_imitation(data, idxs)
            loss_imitation.backward()
            pi_optimizer.step()
        logger.store(LossImitation=loss_imitation.item(), NumItersImitation=n_iter)


    
    start_time = time.time()
    saved_traj_cnt = 0 # counter for the saved experience

    # initialize the clip reward model
    clip_reward_model = CLIPReward(model_clip, device, [env.task_prompt])


    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):

        '''
        # save a video of test
        def test_video():
            pth = os.path.join(save_path, '{}.gif'.format(epoch))
            #if not os.path.exists(pth):
            #    os.mkdir(pth)
            mine_agent.eval() # in eval mode, the actor is also stochastic now
            obs = env.reset()
            gameover = False
            #i = 0
            img_list = []
            while True:
                img_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))
                if gameover:
                    break
                if args.agent_model == 'mineagent':
                    batch = preprocess_obs(obs, device)
                else:
                    batch = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(1,*obs_dim)
                    batch = torch.as_tensor(batch, dtype=torch.float32).to(device)
                with torch.no_grad():
                    act = mine_agent(batch).act
                act = transform_action(act)
                obs, r, gameover, _ = env.step(act)
                #i += 1
            imageio.mimsave(pth, img_list, duration=0.1)
            #env.reset()
            #mine_agent.train()
        '''
        
        # Save model and test
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            #test_video()
            #logger.save_state({'env': env}, None)
            pth = os.path.join(save_path, 'model', 'model_{}.pth'.format(epoch))
            torch.save(mine_agent.state_dict(), pth)


        logger.log('start epoch {}'.format(epoch))
        o, ep_ret, ep_len = env.reset(), 0, 0 # Prepare for interaction with environment
        #clip_reward_model.update_obs(o['rgb_emb']) # preprocess the images embedding
        ep_rewards = []
        ep_obs = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,1,*env.observation_size)
        ep_ret_clip, ep_success, ep_ret_dense = 0, 0, 0
        rgb_list = []
        episode_in_epoch_cnt = 0 # episode id in this epoch

        
        # rollout in the environment
        mine_agent.train() # train mode to sample stochastic actions
        for t in range(local_steps_per_epoch):
            if args.save_raw_rgb:
                rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))

            if args.agent_model == 'mineagent':
                batch_o = preprocess_obs(o, device)
            else:
                batch_o = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,*obs_dim)
                batch_o = torch.as_tensor(batch_o, dtype=torch.float32).to(device)

            with torch.no_grad():
                batch_act = mine_agent.forward_actor_critic(batch_o)
            a, v, logp = batch_act.act, batch_act.val, batch_act.logp
            v = v[0]
            logp = logp[0]
            #print('a,v,logp = ', a, v, logp)

            a_env = transform_action(a)
            next_o, r, d, _ = env.step(a_env)
            success = r

            # update the recent 16 frames, compute intrinsic reward
            #clip_reward_model.update_obs(next_o['rgb_emb'])
            #r_clip = clip_reward_model.reward(mode=args.clip_reward_mode)

            r = r * args.reward_success + args.reward_step # + r_clip * args.reward_clip # weighted sum of different rewards
            ep_rewards.append(r)
            ep_obs = torch.cat((ep_obs, 
                torch_normalize(np.asarray(next_o['rgb'], dtype=np.int)).view(1,1,*env.observation_size)), 1)

            # dense reward
            if args.use_dense:
                r_dense = next_o['dense_reward']
                r += r_dense * args.reward_dense
                ep_ret_dense += r_dense

            ep_success += success
            if ep_success > 1:
                ep_success = 1
            #ep_ret_clip += r_clip
            ep_ret += r
            ep_len += 1

            # save and log
            if args.agent_model == 'mineagent':
                batch_o.to_numpy() # less gpu mem
            else:
                batch_o = batch_o.cpu().numpy()
            buf.store(batch_o, a[0].cpu().numpy(), r, v, logp) # the stored reward will be modified at episode end, if use CLIP reward
            logger.store(VVals=v.detach().cpu().numpy())
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                # compute CLIP embeddings and rewards for each step.
                # modify the trajectory rewards in the buffer
                clip_reward_model.compute_all_embeddings(ep_obs)
                ep_rewards_clip = clip_reward_model.compute_all_rewards()
                #print(len(ep_rewards_clip), len(ep_rewards), ep_obs.shape)
                ep_rewards = np.asarray(ep_rewards) + args.reward_clip * ep_rewards_clip
                ep_ret_clip = np.sum(ep_rewards_clip)
                ep_ret += ep_ret_clip
                buf.modify_trajectory_rewards(ep_rewards)

                # check and add to imitation buffer if the trajectory ends
                if terminal:
                    if args.agent_model == 'mineagent':
                        obs_ = Batch.cat(buf.obs_buf[buf.path_start_idx: buf.ptr])
                    else:
                        obs_ = buf.obs_buf[buf.path_start_idx: buf.ptr].copy()
                    act_ = buf.act_buf[buf.path_start_idx: buf.ptr].copy()
                    if args.save_raw_rgb:
                        rgb_list.append(np.asarray(o['rgb'], dtype=np.uint8))
                    rgb_list = np.asarray(rgb_list)
                    #print(rgb_list.shape)
                    expert_save_dir = os.path.join(args.save_path, 'expert_buffer') if args.save_expert_data else None
                    imitation_buf.eval_and_store(obs_, act_, ep_ret_clip, int(ep_success), rgb_list, expert_save_dir)

                    # save the experience
                    if args.save_all_data:
                        pth = os.path.join(args.save_path, 'experience_buffer', 'traj_{}.pth'.format(saved_traj_cnt))
                        pickle.dump([obs_, act_, ep_ret_clip, int(ep_success), rgb_list], open(pth, 'wb'))
                        saved_traj_cnt += 1

                    # save the gif
                    if args.save_raw_rgb and ((epoch % save_freq == 0) or (epoch == epochs-1)) and episode_in_epoch_cnt==0:
                        pth = os.path.join(args.save_path, 'gif', '{}_ret{}_success{}.gif'.format(epoch, int(ep_ret), int(ep_success)))
                        imageio.mimsave(pth, [np.transpose(i_, [1,2,0]) for i_ in rgb_list], duration=0.1)


                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    if args.agent_model == 'mineagent':
                        batch_o = preprocess_obs(o, device)
                    else:
                        batch_o = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,*obs_dim)
                        batch_o = torch.as_tensor(batch_o, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        v = mine_agent.forward_actor_critic(batch_o).val
                    v = v[0].cpu().detach().numpy()
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpRetClip=ep_ret_clip, EpSuccess=ep_success, 
                        EpRetDense=ep_ret_dense)

                o, ep_ret, ep_len = env.reset(), 0, 0
                ep_ret_clip, ep_success, ep_ret_dense = 0, 0, 0
                ep_rewards = []
                ep_obs = torch_normalize(np.asarray(o['rgb'], dtype=np.int)).view(1,1,*env.observation_size)
                #clip_reward_model.reset() # don't forget to reset the clip images buffer
                #clip_reward_model.update_obs(o['rgb_emb']) # preprocess the images embedding
                rgb_list = []
                episode_in_epoch_cnt += 1


        # Perform PPO update!
        update()
        episode_in_epoch_cnt = 0

        # Perform self-imitation
        if imitation_buf.cur_size >= 1 and (epoch % args.imitate_freq == 0) and epoch > 0:
            for i_imitate in range(args.imitate_epoch):
                update_imitation()
            logger.store(ImitationBufferSuccess=imitation_buf.success_rate,
                ImitationBufferReturn=imitation_buf.avg_return,
                ImitationBufferAcceptReturn=imitation_buf.baseline,
                ImitationBufferNumTraj=imitation_buf.cur_size)
            # Log info about imitation
            logger.log_tabular('LossImitation', average_only=True)
            logger.log_tabular('NumItersImitation', average_only=True)
            logger.log_tabular('ImitationBufferSuccess', average_only=True)
            logger.log_tabular('ImitationBufferReturn', average_only=True)
            logger.log_tabular('ImitationBufferAcceptReturn', average_only=True)
            logger.log_tabular('ImitationBufferNumTraj', average_only=True)
        elif epoch == 0:
            logger.log_tabular('LossImitation', 0)
            logger.log_tabular('NumItersImitation', 0)
            logger.log_tabular('ImitationBufferSuccess', 0)
            logger.log_tabular('ImitationBufferReturn', 0)
            logger.log_tabular('ImitationBufferAcceptReturn', 0)
            logger.log_tabular('ImitationBufferNumTraj', 0)


        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpRetClip', with_min_and_max=True)
        logger.log_tabular('EpSuccess', with_min_and_max=True)
        logger.log_tabular('EpRetDense', with_min_and_max=True)
        logger.log_tabular('EpLen', with_min_and_max=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

        # to avoid destroying too many blocks, remake the environment
        if (epoch % 50 == 0) and epoch>0:
            env.remake_env()
            # save the imitation learning buffer
            #pth = os.path.join(save_path, 'buffer_{}.pth'.format(epoch))
            #pickle.dump(imitation_buf, open(pth, 'wb'))
        