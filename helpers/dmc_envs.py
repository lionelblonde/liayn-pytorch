import gym
from gym import spaces

from dm_control import suite
from dm_env import specs

from dm_control.suite.wrappers import pixels

from helpers.atari_wrappers import TimeLimit, WarpFrame
from helpers.opencv_util import OpenCVImageViewer


# for domain_name, task_name in suite.ALL_TASKS:
#     print(domain_name, task_name)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Core.

def convert_dm_control_to_gym_space(dm_control_space):
    """Convert dm_control space to gym space"""
    if isinstance(dm_control_space, specs.BoundedArray):
        space = spaces.Box(low=dm_control_space.minimum,
                           high=dm_control_space.maximum,
                           dtype=dm_control_space.dtype)
        assert space.shape == dm_control_space.shape
        return space
    elif (isinstance(dm_control_space, specs.Array)
          and not isinstance(dm_control_space, specs.BoundedArray)):
        space = spaces.Box(low=-float('inf'),
                           high=float('inf'),
                           shape=dm_control_space.shape,
                           dtype=dm_control_space.dtype)
        return space
    elif isinstance(dm_control_space, dict):
        space = spaces.Dict({key: convert_dm_control_to_gym_space(value)
                             for key, value in dm_control_space.items()})
        return space


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Create DM Control environments.

class DMSuiteEnv(gym.Env):

    def __init__(self,
                 domain_name,
                 task_name,
                 task_kwargs=None,
                 environment_kwargs=None,
                 visualize_reward=False,
                 use_pixels=False):

        self.env = suite.load(domain_name,
                              task_name,
                              task_kwargs=task_kwargs,
                              environment_kwargs=environment_kwargs,
                              visualize_reward=visualize_reward)

        self.use_pixels = use_pixels
        if self.use_pixels:
            self.env = pixels.Wrapper(self.env, pixels_only=True)

        self.flat_key = 'pixels' if self.use_pixels else 'observations'

        self.metadata = {'render.modes': ['human', 'rgb_array'],
                         'video.frames_per_second': round(1.0/self.env.control_timestep())}

        self.observation_space = convert_dm_control_to_gym_space(self.env.observation_spec())
        self.observation_space = self.observation_space[self.flat_key]

        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec())

        self.viewer = None

    def seed(self, seed):
        return self.env.task.random.seed(seed)

    def step(self, action):
        timestep = self.env.step(action)
        observation = timestep.observation[self.flat_key]
        reward = timestep.reward
        done = timestep.last()
        info = {}
        return observation, reward, done, info

    def reset(self):
        timestep = self.env.reset()
        return timestep.observation[self.flat_key]

    def render(self, mode='human', **kwargs):
        if 'camera_id' not in kwargs:
            kwargs['camera_id'] = 0  # Tracking camera
        frame = self.env.physics.render(**kwargs)
        # default size: width=320, height=240
        if mode == 'rgb_array':
            return frame
        elif mode == 'human':
            if self.viewer is None:
                # At first pass, create the viewer
                self.viewer = OpenCVImageViewer()
            self.viewer.imshow(frame)
            return self.viewer.isopen
        else:
            raise NotImplementedError

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return self.env.close()


def make_dmc(env_id):

    out = env_id.split('-')
    assert len(out) == 4, "invalid environment name"
    domain_name, task_name, inputs, _ = out  # unpack split env name
    assert inputs in ['Feat', 'Pix'], "invalid environment fragment: {}".format(inputs)
    use_pixels = (inputs == 'Pix')
    # Re-format the domain and task names
    if domain_name == 'Humanoid_CMU':
        domain_name = 'humanoid_CMU'
    else:
        domain_name = domain_name.lower()
    env = DMSuiteEnv(domain_name=domain_name,
                     task_name=task_name.lower(),
                     task_kwargs=None,
                     environment_kwargs={'flat_observation': True},
                     visualize_reward=False,
                     use_pixels=use_pixels)
    if use_pixels:
        env = WarpFrame(env, width=64, height=64, grayscale=True)
    env = TimeLimit(env, max_episode_steps=int(1e3))
    return env
