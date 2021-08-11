import random
from math import floor
from collections import defaultdict

import numpy as np

from helpers.math_util import discount
from helpers.misc_util import zipsame
from helpers.segment_tree import SumSegmentTree, MinSegmentTree


class RingBuffer(object):

    def __init__(self, maxlen, shape, dtype='float32'):
        """Ring buffer implementation"""
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape, dtype=dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length
            self.length += 1
            self.data[(self.start + self.length - 1) % self.maxlen] = v

        elif self.length == self.maxlen:
            # No space, "remove" the first item
            self.start = (self.start + 1) % self.maxlen
            self.data[(self.start + self.length - 1) % self.maxlen] = v

        else:
            # This should never happen
            raise RuntimeError()


class ReplayBuffer(object):

    def __init__(self, capacity, shapes):
        self.capacity = capacity
        self.shapes = shapes
        self.ring_buffers = {n: RingBuffer(self.capacity, s) for n, s in self.shapes.items()}

    def batchify(self, idxs):
        """Collect a batch from indices"""
        transitions = {n: array_min2d(self.ring_buffers[n].get_batch(idxs))
                       for n in self.ring_buffers.keys()}
        transitions['idxs'] = idxs  # add idxs too
        return transitions

    def sample(self, batch_size, patcher):
        """Sample transitions uniformly from the replay buffer"""
        idxs = np.random.randint(low=0, high=self.num_entries, size=batch_size)
        transitions = self.batchify(idxs)

        if patcher is not None:
            # Patch the rewards
            transitions['rews'] = patcher(
                transitions['obs0'],
                transitions['acs'],
                transitions['obs1'],
            )

        return transitions

    def sample_recent(self, batch_size, patcher, window):
        """Sample transitions from the most recent ones
        `window_width` designates how far we go back in time
        """
        width = window if window <= self.num_entries else self.num_entries
        # Extract the indices of the 'width' most recent entries
        idxs = np.ones(width) * self.latest_entry_idx
        idxs -= np.arange(start=width - 1, stop=-1, step=-1)
        idxs += self.limit
        idxs %= self.limit
        idxs = idxs.astype(int)
        assert len(idxs) == width
        # Subsample from the isolated indices
        idxs = np.random.choice(idxs, size=batch_size)
        # Collect the transitions associated w/ the sampled indices from the replay buffer
        transitions = self.batchify(idxs)

        if patcher is not None:
            # Patch the rewards
            transitions['rews'] = patcher(
                transitions['obs0'],
                transitions['acs'],
                transitions['obs1'],
            )

        return transitions

    def lookahead(self, transitions, n, gamma, patcher):
        """Perform n-step TD lookahead estimations starting from every transition"""
        assert 0 <= gamma <= 1

        # Initiate the batch of transition data necessary to perform n-step TD backups
        la_batch = defaultdict(list)

        # Iterate over the indices to deploy the n-step backup for each
        for idx in transitions['idxs']:
            # Create indexes of transitions in lookahead of lengths max `n` following sampled one
            la_end_idx = min(idx + n, self.num_entries) - 1
            la_idxs = np.array(range(idx, la_end_idx + 1))
            # Collect the batch for the lookahead rollout indices
            la_transitions = self.batchify(la_idxs)
            if patcher is not None:
                # Patch the rewards
                la_transitions['rews'] = patcher(
                    la_transitions['obs0'],
                    la_transitions['acs'],
                    la_transitions['obs1'],
                )
            # Only keep data from the current episode, drop everything after episode reset, if any
            dones = la_transitions['dones1']
            ep_end_idx = idx + list(dones).index(1.0) if 1.0 in dones else la_end_idx
            la_is_trimmed = 0.0 if ep_end_idx == la_end_idx else 1.0
            # Compute lookahead length
            td_len = ep_end_idx - idx + 1
            # Trim down the lookahead transitions
            la_rews = la_transitions['rews'][:td_len]
            # Compute discounted cumulative reward
            la_discounted_sum_n_rews = discount(la_rews, gamma)[0]
            # Populate the batch for this n-step TD backup
            la_batch['obs0'].append(la_transitions['obs0'][0])
            la_batch['obs1'].append(la_transitions['obs1'][td_len - 1])
            la_batch['acs'].append(la_transitions['acs'][0])
            la_batch['rews'].append(la_discounted_sum_n_rews)
            la_batch['dones1'].append(la_is_trimmed)
            la_batch['td_len'].append(td_len)

            # Add the first next state too in case it is needed for an auxiliary task
            la_batch['obs1_td1'].append(la_transitions['obs1'][0])

            if 'obs0_orig' in la_transitions.keys():
                la_batch['obs0_orig'].append(la_transitions['obs0_orig'][0])
            if 'obs1_orig' in la_transitions.keys():
                la_batch['obs1_orig'].append(la_transitions['obs1_orig'][td_len - 1])
            if 'acs_orig' in la_transitions.keys():
                la_batch['acs_orig'].append(la_transitions['acs_orig'][0])

            # # This block: sanity checker
            # from helpers import logger
            # logger.info("\n\n")
            # print("idx: {}".format(idx))
            # print("la_end_idx: {}".format(la_end_idx))
            # print("la_idxs: {}".format(la_idxs))
            # print("td_len: {}".format(td_len))
            # print("['rews'][:td_len]: {}".format(la_rews))
            # print("la_discounted_sum_n_rews: {}".format(la_discounted_sum_n_rews))
            # print("['obs0']: {}".format(la_transitions['obs0']))
            # print("['obs1']: {}".format(la_transitions['obs1']))
            # print("['dones1']: {}".format(la_transitions['dones1']))
            # print("['obs0'][0]: {}".format(la_transitions['obs0'][0]))
            # print("['obs1'][td_len - 1]: {}".format(la_transitions['obs1'][td_len - 1]))
            # print("la_is_trimmed: {}".format(la_is_trimmed))
            # logger.info("\n\n")

        la_batch['idxs'] = transitions['idxs']

        # Wrap every value with `array_min2d`
        la_batch = {k: array_min2d(v) for k, v in la_batch.items()}
        return la_batch

    def lookahead_sample(self, batch_size, n, gamma, patcher):
        """Sample from the replay buffer.
        This function is for n-step TD backups, where n > 1
        """
        # Sample a batch of transitions
        transitions = self.sample(batch_size, patcher)
        # Expand each transition with a n-step TD lookahead
        return self.lookahead(transitions, n, gamma, patcher)

    def append(self, transition):
        """Add transition to the replay buffer"""
        assert self.ring_buffers.keys() == transition.keys(), "keys must coincide"
        for k in self.ring_buffers.keys():
            self.ring_buffers[k].append(transition[k])

    def __repr__(self):
        return "ReplayBuffer(capacity={})".format(self.capacity)

    @property
    def latest_entry_idx(self):
        # Since all the functions do exactly the same for every RingBuffer, pick arbitrarily
        pick = self.ring_buffers['obs0']
        return (pick.start + pick.length - 1) % pick.maxlen

    @property
    def num_entries(self):
        # Since all the functions do exactly the same for every RingBuffer, pick arbitrarily
        return len(self.ring_buffers['obs0'])


class PrioritizedReplayBuffer(ReplayBuffer):
    """'Prioritized Experience Replay' replay buffer implementation
    Reference: https://arxiv.org/pdf/1511.05952.pdf
    """

    def __init__(self, capacity, shapes,
                 alpha, beta, ranked=False, max_priority=1.0):
        """`alpha` determines how much prioritization is used
        0: none, equivalent to uniform sampling
        1: full prioritization
        `beta` (defined in `__init__`) represents to what degree importance weights are used.
        """
        super(PrioritizedReplayBuffer, self).__init__(capacity, shapes)
        assert 0. <= alpha <= 1.
        assert beta > 0, "beta must be positive"
        self.alpha = alpha
        self.beta = beta
        self.max_priority = max_priority
        # Calculate the segment tree capacity suited to the user-specified capacity
        self.st_cap = segment_tree_capacity(capacity)
        # Create segment tree objects as data collection structure for priorities.
        # It provides an efficient way of calculating a cumulative sum of priorities
        self.sum_st = SumSegmentTree(self.st_cap)  # with `operator.add` operation
        self.min_st = MinSegmentTree(self.st_cap)  # with `min` operation
        # Whether it is the ranked version or not
        self.ranked = ranked
        if self.ranked:
            # Create a dict that will contain all the (index, priority) pairs
            self.i_p = {}

    def _sample_w_priorities(self, batch_size):
        """Sample in proportion to priorities, implemented w/ segment tree.
        This function samples a batch of transitions indices, directly from the priorities.
        Segment trees enable the emulation of a categorical sampling process by
        relying on what they shine at: computing cumulative sums.
        Imagine a stack of blocks.
        Each block (transition) has a height equal to its priority.
        The total height therefore is the sum of all priorities, `p_total`.
        `u_sample` is sampled from a U[0,1]. `u_sample * p_total` consequently is
        a value uniformly sampled from U[0,p_total], a height on the stacked blocks.
        `find_prefixsum_idx` returns the highest index (block id) such that the sum of
        preceeding priorities (block heights) is <= to the uniformly sampled height.
        The process is equivalent to sampling from a categorical distribution over
        the transitions (it might even be how some library implement categorical sampling).
        Since the height is sampled uniformly, the prob of landing in a block is proportional
        to the height of said block. The height being the priority value, the higher the
        priority, the higher the prob of being selected.
        """
        assert self.num_entries > 1, "segment tree ends with capacity-1, must be >0"
        transition_idxs = []
        # Sum the priorities of the transitions currently in the buffer
        p_total = self.sum_st.sum(end=self.num_entries - 1)
        # `start` is 0 by default, `end` is length - 1
        # Divide equally into `batch_size` ranges (appendix B.2.1)
        p_pieces = p_total / batch_size
        # Sample `batch_size` samples independently, each from within the associated range
        # which is referred to as 'stratified sampling'
        for i in range(batch_size):
            # Sample a value uniformly from the unit interval
            unit_u_sample = random.random()  # ~U[0,1]
            # Scale and shift the sampled value to be within the range of cummulative priorities
            u_sample = (unit_u_sample * p_pieces) + (i * p_pieces)
            # Retrieve the transition index associated with `u_sample`
            # i.e. in which block the sample landed
            transition_idx = self.sum_st.find_prefixsum_idx(u_sample)
            transition_idxs.append(transition_idx)
        return np.array(transition_idxs)

    def _sample(self, batch_size, sampling_fn, patcher):
        """Sample from the replay buffer according to assigned priorities
        while using importance weights to offset the biasing effect of non-uniform sampling.
        `beta` (defined in `__init__`) represents to what degree importance weights are used.
        """
        # Sample transition idxs according to the samplign function
        idxs = sampling_fn(batch_size=batch_size)

        # Initialize importance weights
        iws = []
        # Create var for lowest sampling prob among transitions currently in the buffer,
        # equal to lowest priority divided by the sum of all priorities
        lowest_prob = self.min_st.min(end=self.num_entries) / self.sum_st.sum(end=self.num_entries)
        # Create for maximum weight var for weight scaling purposes (eq in 3.4. PER paper)
        max_weight = (self.num_entries * lowest_prob) ** (-self.beta)

        # Create a weight for every selected transition
        for idx in idxs:
            # Compute the probability assigned to the transition
            prob_transition = self.sum_st[idx] / self.sum_st.sum(end=self.num_entries)
            # Compute the transition weight
            weight_transition = (self.num_entries * prob_transition) ** (-self.beta)
            iws.append(weight_transition / max_weight)

        # Collect batch of transitions w/ iws and indices
        w_transitions = super().batchify(idxs)
        w_transitions['iws'] = array_min2d(np.array(iws))
        w_transitions['idxs'] = np.array(idxs)

        if patcher is not None:
            # Patch the rewards
            w_transitions['rews'] = patcher(
                w_transitions['obs0'],
                w_transitions['acs'],
                w_transitions['obs1'],
            )

        return w_transitions

    def sample(self, batch_size, patcher):
        return self._sample(batch_size, self._sample_w_priorities, patcher)

    def sample_uniform(self, *args):
        return super().sample(*args)

    def lookahead_sample(self, batch_size, n, gamma, patcher):
        """Sample from the replay buffer according to assigned priorities.
        This function is for n-step TD backups, where n > 1
        """
        assert n > 1
        # Sample a batch of transitions
        transitions = self.sample(batch_size, patcher)
        # Expand each transition w/ a n-step TD lookahead
        la_batch = super().lookahead(transitions, n, gamma, patcher)
        # Add iws to the dict
        la_batch['iws'] = transitions['iws']
        return la_batch

    def append(self, *args, **kwargs):
        super().append(*args, **kwargs)
        idx = self.latest_entry_idx
        # Assign highest priority value to newly added elements (line 6 alg PER paper)
        self.sum_st[idx] = self.max_priority ** self.alpha
        self.min_st[idx] = self.max_priority ** self.alpha

    def update_priorities(self, idxs, priorities):
        """Update priorities according to the PER paper, i.e. by updating
        only the priority of sampled transitions. A priority priorities[i] is
        assigned to the transition at index indices[i].
        Note: not in use in the vanilla setting, but here if needed in extensions.
        """
        if self.ranked:
            # Override the priorities to be 1 / (rank(priority) + 1)
            # Add new index, priority pairs to the list
            self.i_p.update({i: p for i, p in zipsame(idxs, priorities)})
            # Rank the indices by priorities
            i_sorted_by_p = sorted(self.i_p.items(), key=lambda t: t[1], reverse=True)
            # Create the index, rank dict
            i_r = {i: i_sorted_by_p.index((i, p)) for i, p in self.i_p.items()}
            # Unpack indices and ranks
            _idxs, ranks = zipsame(*i_r.items())
            # Override the indices and priorities
            idxs = list(_idxs)
            priorities = [1. / (rank + 1) for rank in ranks]  # start ranks at 1

        assert len(idxs) == len(priorities), "the two arrays must be the same length"
        for idx, priority in zipsame(idxs, priorities):
            assert priority > 0, "priorities must be positive"
            assert 0 <= idx < self.num_entries, "no element in buffer associated w/ index"
            self.sum_st[idx] = priority ** self.alpha
            self.min_st[idx] = priority ** self.alpha
            # Update max priority currently in the buffer
            self.max_priority = max(priority, self.max_priority)

        if self.ranked:
            # Return indices and associated overriden priorities
            # Note: returned values are only used in the UNREAL priority update function
            return idxs, priorities

    def __repr__(self):
        fmt = "PrioritizedReplayBuffer(capacity={}, alpha={}, beta={}, "
        fmt += "ranked={}, max_priority={})"
        return fmt.format(self.capacity, self.alpha, self.beta,
                          self.ranked, self.max_priority)


class UnrealReplayBuffer(PrioritizedReplayBuffer):
    """'Reinforcement Learning w/ unsupervised Auxiliary Tasks' replay buffer implementation
    Reference: https://arxiv.org/pdf/1611.05397.pdf
    """

    def __init__(self, capacity, shapes, max_priority=1.0):
        """Reuse of the 'PrioritizedReplayBuffer' constructor w/:
            - `alpha` arbitrarily set to 1. (unused)
            - `beta` arbitrarily set to 1. (unused)
            - `ranked` set to True (necessary to have access to the ranks)
        """
        super(UnrealReplayBuffer, self).__init__(capacity, shapes,
                                                 1., 1, True, max_priority)
        # Create two extra `SumSegmentTree` objects: one for 'bad' transitions, one for 'good' ones
        self.b_sum_st = SumSegmentTree(self.st_cap)  # with `operator.add` operation
        self.g_sum_st = SumSegmentTree(self.st_cap)  # with `operator.add` operation

    def _sample_unreal(self, batch_size):
        """Sample uniformly from which virtual sub-buffer to pick: bad or good transitions,
        then sample uniformly a transition from the previously picked virtual sub-buffer.
        Implemented w/ segment tree.
        Since `b_sum_st` and `g_sum_st` contain priorities in {0,1}, sampling according to
        priorities samples a transition uniformly from the transitions having priority 1
        in the previously sampled virtual sub-buffer (bad or good transitions).
        Note: the priorities were used (in `update_priorities`) to determine in which
        virtual sub-buffer a transition belongs.
        """
        factor = 5  # hyperparam?
        assert factor > 1
        assert self.num_entries
        transition_idxs = []
        # Sum the priorities (in {0,1}) of the transitions currently in the buffers
        # Since values are in {0,1}, the sums correspond to cardinalities (#b and #g)
        b_p_total = self.b_sum_st.sum(end=self.num_entries)
        g_p_total = self.g_sum_st.sum(end=self.num_entries)
        # `start` is 0 by default, `end` is length - 1 (classic python)
        # Ensure no repeats once the number of memory entries is high enough
        no_repeats = self.num_entries > factor * batch_size
        for _ in range(batch_size):
            while True:
                # Sample a value uniformly from the unit interval
                unit_u_sample = random.random()  # ~U[0,1]
                # Sample a number in {0,1} to decide whether to sample a b/g transition
                is_g = np.random.randint(2)
                p_total = g_p_total if is_g else b_p_total
                # Scale the sampled value to the total sum of priorities
                u_sample = unit_u_sample * p_total
                # Retrieve the transition index associated with `u_sample`
                # i.e. in which block the sample landed
                b_o_g_sum_st = self.g_sum_st if is_g else self.b_sum_st
                transition_idx = b_o_g_sum_st.find_prefixsum_idx(u_sample)
                if not no_repeats or transition_idx not in transition_idxs:
                    transition_idxs.append(transition_idx)
                    break
        return np.array(transition_idxs)

    def sample(self, batch_size, patcher):
        return super()._sample(batch_size, self._sample_unreal, patcher)

    def lookahead_sample(self, *args):
        return super().lookahead_sample(*args)

    def append(self, *args, **kwargs):
        super().append(*args, **kwargs)
        idx = self.latest_entry_idx
        # Add newly added elements to 'good' and 'bad' virtual sub-buffer
        self.b_sum_st[idx] = 1
        self.g_sum_st[idx] = 1

    def update_priorities(self, idxs, priorities):
        # Update priorities via the legacy method, used w/ ranked approach
        idxs, priorities = super().update_priorities(idxs, priorities)
        # Register whether a transition is b/g in the UNREAL-specific sum trees
        for idx, priority in zipsame(idxs, priorities):
            # Decide whether the transition to be added is good or bad
            # Get the rank from the priority
            # Note: UnrealReplayBuffer inherits from PER w/ 'ranked' set to True
            rank = (1. / priority) - 1
            thres = floor(.5 * self.num_entries)
            is_g = rank < thres
            is_g *= 1  # HAXX: multiply by 1 to cast the bool into an int
            # Fill the good and bad sum segment trees w/ the obtained value
            self.b_sum_st[idx] = 1 - is_g
            self.g_sum_st[idx] = is_g

    def __repr__(self):
        fmt = "UnrealReplayBuffer(capacity={}, max_priority={})"
        return fmt.format(self.capacity, self.max_priority)


def segment_tree_capacity(capacity):
    """Using a Segment Tree data structure imposes capacity being a power of 2.
    This function finds the highest power of 2 below the user-specified capacity.
    """
    st_cap = 1
    while st_cap < capacity:
        # if the current tree capacity is not past the user-specified one, continue
        st_cap *= 2
    return st_cap


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)
