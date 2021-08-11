from collections import defaultdict
import os.path as osp
import glob
import h5py
import numpy as np
import torch

from helpers import logger


def save_dict_h5py(data, fname):
    """Save dictionary containing numpy arrays to h5py file."""
    with h5py.File(fname, 'w') as hf:
        for key in data.keys():
            hf.create_dataset(key, data=data[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    data = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            data[key] = hf[key][()]
    return data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        return {k: v[i, ...].astype(np.float32) for k, v in self.data.items()}

    def __len__(self):
        return list(self.data.values())[0].shape[0]


class DemoDataset(Dataset):

    def __init__(self, expert_path, num_demos, env, wrap_absorb):
        logger.info("dataset:")
        logger.info(f"expert path: {expert_path}")
        logger.info(f"num_demos: {num_demos}")
        self.num_demos = num_demos
        self.data = defaultdict(list)
        self.stats = defaultdict(list)
        logger.info(">>>> loading demos")
        # Go over the demos, sorted in alphabetical order
        for i, f in enumerate(sorted(glob.glob(osp.join(expert_path, "*.h5")))):
            # If the desired number of demos has been aggregated, leave
            if i == num_demos:
                break
            # Log the location of the loaded demo
            logger.info("[DEMO DATASET] demo #{} loaded from: {}".format(str(i).zfill(3), f))
            # Load the demo from the file
            tmp = load_dict_h5py(f)
            # Remove undesirable keys (at least in this application)
            assert tmp['dones1'][-1], "by construction"  # making sure every ep ends with done
            tmp.pop('pix_obs0', None)
            tmp.pop('pix_obs1', None)
            # Extract and display content dims
            dims = {k: tmp[k].shape[1:] for k in tmp.keys()}
            dims = ' | '.join(["{}={}".format(k, v) for k, v in dims.items()])
            logger.info("[DEMO DATASET]      dims: {}".format(dims))
            # Get episode statistics
            ep_len = tmp.pop('ep_lens', None)  # return and delete key
            ep_ret = tmp.pop('ep_env_rets', None)  # return and delete key
            assert isinstance(ep_len, np.int64), "each file should contain only one episode"
            assert isinstance(ep_ret, np.float64), "each file should contain only one episode"
            logger.info("[DEMO DATASET]      {}{}".format("ep_len".ljust(20, '-'), ep_len))
            logger.info("[DEMO DATASET]      {}{}".format("ep_ret".ljust(20, '-'), ep_ret))
            self.stats['ep_len'].append(ep_len)
            self.stats['ep_ret'].append(ep_ret)
            # Determine if terminal because of timeout or real termination
            terminal = not ep_len == env._max_episode_steps

            # Subsample trajectory
            sub_rate = 20  # N=20 in the original GAIL paper
            start = np.random.randint(low=0, high=sub_rate)
            indices = [start + (i * sub_rate) for i in range(ep_len // sub_rate)]
            ep_len = len(indices)  # overwrite ep_len
            logger.info("[DEMO DATASET]      {}{}".format("subsample".ljust(20, '-'),
                                                          "{}(sub_rate={})".format(ep_len, sub_rate)))
            for k in tmp.keys():
                tmp[k] = tmp[k][indices]

            # Collect the demo's content
            if wrap_absorb:
                if tmp['dones1'][-1] and terminal:
                    # If the last subsampled transition is done, then it must be
                    # the very last transition of the episode, and testing whether it is
                    # a true terminal state is given by 'terminal' determined above.
                    logger.info("[DEMO DATASET] >>>> wrapping with absorbing transition <<<<")
                    # Wrap with an absorbing state
                    obs0 = np.concatenate(
                        [tmp['obs0'],
                         np.zeros((ep_len, 1))],
                        axis=-1
                    )
                    acs = np.concatenate(
                        [tmp['acs'],
                         np.zeros((ep_len, 1))],
                        axis=-1
                    )
                    obs1 = np.concatenate(
                        [tmp['obs1'],
                         np.concatenate(
                            [np.zeros((ep_len - 1, 1)),
                             np.ones((1, 1))],
                            axis=0)],
                        axis=-1
                    )
                    # Add absorbing transition
                    obs0 = np.concatenate(
                        [obs0,
                         np.expand_dims(np.append(np.zeros_like(tmp['obs0'][-1]), 1), axis=0)],
                        axis=0
                    )
                    acs = np.concatenate(
                        [acs,
                         np.expand_dims(np.append(np.zeros_like(tmp['acs'][-1]), 1), axis=0)],
                        axis=0
                    )
                    obs1 = np.concatenate(
                        [obs1,
                         np.expand_dims(np.append(np.zeros_like(tmp['obs1'][-1]), 1), axis=0)],
                        axis=0
                    )
                    self.data['obs0'].append(obs0)
                    self.data['acs'].append(acs)
                    self.data['obs1'].append(obs1)
                else:
                    self.data['obs0'].append(np.concatenate([tmp['obs0'],
                                                             np.zeros((ep_len, 1))], axis=-1))
                    self.data['acs'].append(np.concatenate([tmp['acs'],
                                                            np.zeros((ep_len, 1))], axis=-1))
                    self.data['obs1'].append(np.concatenate([tmp['obs1'],
                                                             np.zeros((ep_len, 1))], axis=-1))

                self.data['obs0_orig'].append(tmp['obs0'])

            else:
                self.data['obs0'].append(tmp['obs0'])
                self.data['acs'].append(tmp['acs'])
                self.data['obs1'].append(tmp['obs1'])

        # Transform structures into arrays
        for k, v in self.stats.items():
            self.stats[k] = np.array(v)
        for k, v in self.data.items():
            self.data[k] = np.concatenate(v, axis=0)

        # Log demos' statistics
        logger.info("[DEMO DATASET] keys extracted: {}".format(list(self.data.keys())))
        lens, rets = self.stats['ep_len'], self.stats['ep_ret']
        logger.info("[DEMO DATASET] got {} transitions, from {} eps".format(len(self), self.num_demos))
        logger.info("[DEMO DATASET] episodic length: {}({})".format(np.mean(lens), np.std(lens)))
        logger.info("[DEMO DATASET] episodic return: {}({})".format(np.mean(rets), np.std(rets)))
