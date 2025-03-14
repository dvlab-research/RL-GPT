import minedojo
import sys
#import imageio
import numpy as np
import time

# reset() bug fixed
# use the multi-discrete action space (3,3,4,25,25,8). For the last dim, allow 0,1,3 only
# further tune and clip the action space, modify transform_action(). 22/9/1

from mineagent.batch import Batch
import torch
from mineclip_official import torch_normalize
from mineagent.features.voxel.flattened_voxel_block import VOXEL_BLOCK_NAME_MAP

def preprocess_obs(obs, device):
    """
    Here you preprocess the raw env obs to pass to the agent.
    Preprocessing includes, for example, use MineCLIP to extract image feature and prompt feature,
    flatten and embed voxel names, mask unused obs, etc.
    """
    B = 1

    def cvt_voxels(vox):
        ret = np.zeros(3*3*3, dtype=np.long)
        for i, v in enumerate(vox.reshape(3*3*3)):
            if v in VOXEL_BLOCK_NAME_MAP:
                ret[i] = VOXEL_BLOCK_NAME_MAP[v]
        return ret

    # I consider the move and functional action only, because the camera space is too large?
    # construct a 3*3*4*3 action embedding
    def cvt_action(act):
        if act[5]<=1:
            return act[0] + 3*act[1] + 9*act[2] + 36*act[5]
        elif act[5]==3:
            return act[0] + 3*act[1] + 9*act[2] + 72
        else:
            raise Exception('Action[5] should be 0,1,3')

    yaw_ = np.deg2rad(obs["location_stats"]["yaw"])
    pitch_ = np.deg2rad(obs["location_stats"]["pitch"])
    obs_ = {
        "compass": torch.as_tensor([np.concatenate([np.cos(yaw_), np.sin(yaw_), np.cos(pitch_), np.sin(pitch_)])], device=device),
        "gps": torch.as_tensor([obs["location_stats"]["pos"]], device=device),
        "voxels": torch.as_tensor(
            [cvt_voxels(obs["voxels"]["block_name"])], dtype=torch.long, device=device
        ),
        "biome_id": torch.tensor(
            [int(obs["location_stats"]["biome_id"])], dtype=torch.long, device=device
        ),
        "prev_action": torch.tensor(
            [cvt_action(obs["prev_action"])], dtype=torch.long, device=device
        ),
        "prompt": torch.as_tensor(obs["rgb_emb"], device=device).view(B, 512), 
        # this is actually the image embedding, not prompt embedding (for single task)
    }
    #print(obs_["prev_action"])
    #print(obs_["compass"], yaw_, pitch_)

    #print(Batch(obs=obs_))
    return Batch(obs=obs_)



# Map agent action to env action.

# for [3,3,4,25,25,8] agent action space
'''
def transform_action(action):

    assert action.ndim == 2 # (1, 6)
    action = action[0]
    action = action.cpu().numpy()
    if action[-1] != 0 or action[-1] != 1 or action[-1] != 3:
        action[-1] = 0
    action = np.concatenate([action, np.array([0, 0])])
    return action #(8)
'''

# [56, 3] agent action space as I initially implemented
'''
def transform_action(act):
    assert act.ndim == 2 # (1, 2)
    act = act[0]
    act = act.cpu().numpy()
    act1, act2 = act[0], act[1]
    
    action = [0,0,0,12,12,0,0,0] #self.base_env.action_space.no_op()
    assert act1 < 56
    if act1 == 0: # no op
        action = action
    elif act1 < 3: # forward backward
        action[0] = act1
    elif act1 < 5: # left right
        action[1] = act1 - 2
    elif act1 < 8: # jump sneak sprint
        action[2] = act1 - 4
    elif act1 < 20: # camera pitch 0~11
        action[3] = act1 - 8
    elif act1 < 32: # camera pitch 13~24
        action[3] = act1 - 8 + 1
    elif act1 < 44: # camera yaw 0~11
        action[4] = act1 - 32
    else: # camera yaw 13~24
        action[4] = act1 - 32 + 1

    assert act2 < 3
    if act2 == 1: # use
        action[5] = 1
    elif act2 == 2: #attack
        action[5] = 3
    return action #(8)
'''

# for [3,3,4,5,3] action space, preserve only 5 camera choices 
'''
def transform_action(act):
    assert act.ndim == 2 # (1, 5)
    act = act[0]
    act = act.cpu().numpy()
    
    action = [act[0],act[1],act[2],12,12,0,0,0] #self.base_env.action_space.no_op()

    # no_op, use, attack
    act_use = act[4]
    if act_use == 2:
        act_use = 3
    action[5] = act_use

    # no_op, 2 pitch, 2 yaw
    act_cam = act[3]
    if act_cam == 1:
        action[3] = 11
    elif act_cam == 2:
        action[3] = 13
    elif act_cam == 3:
        action[4] = 11
    elif act_cam == 4:
        action[4] = 13

    #print(action)

    return action #(8)
'''

# [12, 3] action space, 1 choice among walk, jump and camera
# preserve 5 camera actions

def transform_action(act):
    assert act.ndim == 2 # (1, 2)
    act = act[0]
    act = act.cpu().numpy()
    act1, act2 = act[0], act[1]
    
    action = [0,0,0,12,12,0,0,0] #self.base_env.action_space.no_op()
    assert act1 < 12
    if act1 == 0: # no op
        action = action
    elif act1 < 3: # forward backward
        action[0] = act1
    elif act1 < 5: # left right
        action[1] = act1 - 2
    elif act1 < 8: # jump sneak sprint
        action[2] = act1 - 4
    elif act1 == 8: # camera pitch 10
        action[3] = 10
    elif act1 == 9: # camera pitch 14
        action[3] = 14
    elif act1 == 10: # camera yaw 10
        action[4] = 10
    elif act1 == 11: # camera yaw 14
        action[4] = 14

    assert act2 < 3
    if act2 == 1: # use
        action[5] = 1
    elif act2 == 2: #attack
        action[5] = 3
    return action #(8)


'''
9/6 
To support dense reward, you should insert these codes
    for key in kwargs:
        if key in task_specs:
            task_specs.pop(key)
into your MineDojo package minedojo/tasks/_init__.py line 494, before calling '_meta_task_make'

'''
from collections import deque
class MinecraftEnv:

    def __init__(self, task_id, image_size=(160, 256), max_step=500, clip_model=None, device=None, seed=0,
        dense_reward=False, target_name='cow',  biome=None, **kwargs):
        self.observation_size = (3, *image_size)
        self.action_size = 8
        self.dense_reward = dense_reward
        self.biome = biome
        self.max_step = max_step
        self.cur_step = 0
        self.clip_model = clip_model # use mineclip model to precompute embeddings
        self.device = device
        self.seed = seed
        self.task_id = task_id
        self.image_size = image_size
        self.kwargs = kwargs
        self.remake_env()
        self.task_prompt = self.base_env.task_prompt
        self._reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]

    def __del__(self):
        if hasattr(self, 'base_env'):
            self.base_env.close()

    def remake_env(self):
        '''
        call this to reset all the blocks and trees
        should modify line 479 in minedojo/tasks/__init__.py, deep copy the task spec dict:
            import deepcopy
            task_specs = copy.deepcopy(ALL_TASKS_SPECS[task_id])
        '''
        if hasattr(self, 'base_env'):
            self.base_env.close()
        if not self.dense_reward:
            self.base_env = minedojo.make(task_id=self.task_id, image_size=self.image_size, seed=self.seed, specified_biome=self.biome, **self.kwargs)
        else:
            self.base_env = minedojo.make(task_id=self.task_id, image_size=self.image_size, seed=self.seed, specified_biome=self.biome, 
                use_lidar=True, lidar_rays=[
                (np.pi * pitch / 180, np.pi * yaw / 180, 999)
                for pitch in np.arange(-30, 30, 6)
                for yaw in np.arange(-60, 60, 10)], **self.kwargs)
            #self._target_name = target_name
            self._consecutive_distances = deque(maxlen=2)
            self._distance_min = np.inf
        self._first_reset = True
        print('Environment remake: reset all the destroyed blocks!')


    def reset(self):
        if not self._first_reset:
            for cmd in self._reset_cmds:
                self.base_env.unwrapped.execute_cmd(cmd)
            self.base_env.unwrapped.set_time(6000)
            self.base_env.unwrapped.set_weather("clear")
        self._first_reset = False
        self.prev_action = self.base_env.action_space.no_op()

        obs = self.base_env.reset()
        self.cur_step = 0

        if self.clip_model is not None:
            with torch.no_grad():
                img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(1,1,*self.observation_size)
                img_emb = self.clip_model.image_encoder(torch.as_tensor(img,dtype=torch.float).to(self.device))
                obs['rgb_emb'] = img_emb.cpu().numpy() # (1,1,512)
                #print(obs['rgb_emb'])
                obs['prev_action'] = self.prev_action

        if self.dense_reward:
            self._consecutive_distances.clear()
            self._distance_min = np.inf
            entity_in_sight, distance = self._find_distance_to_entity_if_in_sight(obs)
            if entity_in_sight:
                distance = self._distance_min = min(distance, self._distance_min)
                self._consecutive_distances.append(distance)
            else:
                self._consecutive_distances.append(0)

        return obs

    def step(self, act):
        obs, reward, done, info = self.base_env.step(act)
        self.cur_step += 1
        if self.cur_step >= self.max_step:
            done = True
        
        if self.clip_model is not None:
            with torch.no_grad():
                img = torch_normalize(np.asarray(obs['rgb'], dtype=np.int)).view(1,1,*self.observation_size)
                img_emb = self.clip_model.image_encoder(torch.as_tensor(img,dtype=torch.float).to(self.device))
                obs['rgb_emb'] = img_emb.cpu().numpy() # (1,1,512)
                #print(obs['rgb_emb'])
                obs['prev_action'] = self.prev_action

        self.prev_action = act # save the previous action for the agent's observation

        if self.dense_reward:
            entity_in_sight, distance = self._find_distance_to_entity_if_in_sight(obs)
            nav_reward = 0
            if entity_in_sight:
                distance = self._distance_min = min(distance, self._distance_min)
                self._consecutive_distances.append(distance)
                nav_reward = self._consecutive_distances[0] - self._consecutive_distances[1]
                #print('dense:', nav_reward, self._consecutive_distances[1])
            nav_reward = max(0, nav_reward)
            obs['dense_reward'] = nav_reward

        return  obs, reward, done, info

    # for dense reward, find the nearest target in sight
    def _find_distance_to_entity_if_in_sight(self, obs):
        assert self.dense_reward is True
        in_sight, min_distance = False, None
        entities, distances = (
            obs["rays"]["entity_name"],
            obs["rays"]["entity_distance"],
        )
        entity_idx = np.where(entities == self._target_name)[0]
        if len(entity_idx) > 0:
            in_sight = True
            min_distance = np.min(distances[entity_idx])
        return in_sight, min_distance



import habitat
'''
Oct 29
env for multi-process
1. the init function receives a single args
2. not contain CLIP model
3. specially: auto reset an env if done, because all the envs are stepped simultaneously
'''
class MinecraftEnvMP(habitat.RLEnv):

    # def __init__(self, task_id, image_size=(160, 256), max_step=500, clip_model=None, device=None, seed=0,
    #              dense_reward=False, target_name='cow'):
    def __init__(self, args):
        self.args = args
        self.observation_size = (3, *args.image_size)
        self.action_size = 8
        #self.dense_reward = bool(args.use_dense)
        if 'biome' in args:
            self._env = minedojo.make(task_id=args.task_id, image_size=args.image_size, seed=args.seed_env, specified_biome=args.biome)
        else:
            self._env = minedojo.make(task_id=args.task_id, image_size=args.image_size, seed=args.seed_env)

        self.max_step = args.horizon
        self.cur_step = 0
        self.task_prompt = self._env.task_prompt
        #self.clip_model = None  # use mineclip model to precompute embeddings
        #self.device = args.device
        self.seed_env = args.seed_env
        self.task_id = args.task_id
        self.image_size = args.image_size
        #self.number_of_episodes = 10000
        self._first_reset = True
        self._reset_cmds = ["/kill @e[type=!player]",
                            "/clear", "/kill @e[type=item]"]

        self.number_of_episodes = 10000

    def __del__(self):
        if hasattr(self, '_env'):
            self._env.close()

    # auto reset after remake
    def remake(self):
        '''
        call this to reset all the blocks and trees
        should modify line 479 in minedojo/tasks/__init__.py, deep copy the task spec dict:
            import deepcopy
            task_specs = copy.deepcopy(ALL_TASKS_SPECS[task_id])
        '''
        self._env.close()
        if 'biome' in self.args:
            self._env = minedojo.make(task_id=self.task_id, image_size=self.image_size, seed=self.seed_env, specified_biome=self.args.biome)
        else:
            self._env = minedojo.make(task_id=self.task_id, image_size=self.image_size, seed=self.seed_env)
    
        self._first_reset = True
        print('Environment remake: reset all the destroyed blocks!')
        return self._env.reset()

    def reset(self):
        if not self._first_reset:
            for cmd in self._reset_cmds:
                self._env.unwrapped.execute_cmd(cmd)
            self._env.unwrapped.set_time(6000)
            self._env.unwrapped.set_weather("clear")
        self._first_reset = False
        self.prev_action = self._env.action_space.no_op()

        obs = self._env.reset()
        self.cur_step = 0

        obs['prev_action'] = self.prev_action

        return obs

    def step(self, act):
        #print(act)
        obs, reward, done, info = self._env.step(act['action'])
        self.cur_step += 1
        if self.cur_step >= self.max_step:
            done = True

        obs['prev_action'] = self.prev_action
        self.prev_action = np.asarray(act['action'])  # save the previous action for the agent's observation

        return obs, reward, done, info


if __name__ == '__main__':
    #print(minedojo.ALL_TASKS_SPECS)
    env = MinecraftEnv(
        task_id="harvest_milk_with_empty_bucket_and_cow",
        image_size=(160, 256),
    )
    reset_cmds = ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]
    obs = env.reset()
    #print(obs.shape, obs.dtype)
    for t in range (100):
        for i in range(12):
            act = [42,0] #cam
            obs, reward, done, info = env.step(act)
            time.sleep(0.2)
        print('reset')
        for cmd in reset_cmds:
            env.base_env.execute_cmd(cmd)
        obs = env.reset()
'''

#print(env.base_env.task_prompt, env.base_env.task_guidance)
obs = env.reset()
print(obs.shape, obs.dtype)
for t in range (1000):

    #act = env.base_env.action_space.no_op()
    
    if t < 50:
        act = [1,0]   # forward
    elif t < 200:
        act = [0,0]    # stall
    elif t < 400:
        act = [1,2] # attack
    elif t<600:
        act = [50,1] #cam
    elif t<800:
        act = [40,1]
    else:
        act = [1,0]
    
    obs, reward, done, info = env.step(act)
    print(act, reward)

    #print(reward, done, info)

'''