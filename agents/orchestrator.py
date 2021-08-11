import time
from copy import deepcopy
import os
import os.path as osp
from collections import defaultdict
import signal

import wandb
import numpy as np

from helpers import logger
# from helpers.distributed_util import sync_check
from helpers.console_util import timed_cm_wrapper, log_iter_info
from helpers.opencv_util import record_video


debug_lvl = os.environ.get('DEBUG_LVL', 0)
try:
    debug_lvl = np.clip(int(debug_lvl), a_min=0, a_max=3)
except ValueError:
    debug_lvl = 0
DEBUG = bool(debug_lvl >= 1)


def rl_rollout_generator(env, agent, rollout_len):

    t = 0
    # Reset agent's noise process
    agent.reset_noise()
    # Reset agent's env
    ob = np.array(env.reset())

    while True:

        # Predict action
        ac = agent.predict(ob, apply_noise=True)
        # NaN-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)

        if t > 0 and t % rollout_len == 0:
            yield

        # Interact with env(s)
        new_ob, rew, done, _ = env.step(ac)

        # Assemble and store transition in memory
        transition = {
            "obs0": ob,
            "acs": ac,
            "obs1": new_ob,
            "rews": rew,
            "dones1": done,
        }
        agent.store_transition(transition)

        # Set current state with the next
        ob = np.array(deepcopy(new_ob))

        if done:
            # Reset agent's noise process
            agent.reset_noise()
            # Reset agent's env
            ob = np.array(env.reset())

        t += 1


def il_rollout_generator(env, agent, rollout_len):

    t = 0
    # Reset agent's noise process
    agent.reset_noise()
    # Reset agent's env
    ob = np.array(env.reset())

    while True:

        # Predict action
        ac = agent.predict(ob, apply_noise=True)
        # NaN-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)

        if t > 0 and t % rollout_len == 0:
            yield

        # Interact with env(s)
        new_ob, _, done, _ = env.step(ac)

        if agent.hps.wrap_absorb:
            _ob = np.append(ob, 0)
            _ac = np.append(ac, 0)
            if done and not env._elapsed_steps == env._max_episode_steps:
                # Wrap with an absorbing state
                _new_ob = np.append(np.zeros(agent.ob_shape), 1)
                _rew = agent.get_syn_rew(_ob[None], _ac[None], _new_ob[None])
                _rew = np.asscalar(_rew.cpu().numpy().flatten())
                transition = {
                    "obs0": _ob,
                    "acs": _ac,
                    "obs1": _new_ob,
                    "rews": _rew,
                    "dones1": done,
                    "obs0_orig": ob,
                    "acs_orig": ac,
                    "obs1_orig": new_ob,
                }
                agent.store_transition(transition)
                # Add absorbing transition
                _ob_a = np.append(np.zeros(agent.ob_shape), 1)
                _ac_a = np.append(np.zeros(agent.ac_shape), 1)
                _new_ob_a = np.append(np.zeros(agent.ob_shape), 1)
                _rew_a = agent.get_syn_rew(_ob_a[None], _ac_a[None], _new_ob_a[None])
                _rew_a = np.asscalar(_rew_a.cpu().numpy().flatten())
                transition_a = {
                    "obs0": _ob_a,
                    "acs": _ac_a,
                    "obs1": _new_ob_a,
                    "rews": _rew_a,
                    "dones1": done,
                    "obs0_orig": ob,  # from previous transition, with reward eval on absorbing
                    "acs_orig": ac,  # from previous transition, with reward eval on absorbing
                    "obs1_orig": new_ob,  # from previous transition, with reward eval on absorbing
                }
                agent.store_transition(transition_a)
            else:
                _new_ob = np.append(new_ob, 0)
                _rew = agent.get_syn_rew(_ob[None], _ac[None], _new_ob[None])
                _rew = np.asscalar(_rew.cpu().numpy().flatten())
                transition = {
                    "obs0": _ob,
                    "acs": _ac,
                    "obs1": _new_ob,
                    "rews": _rew,
                    "dones1": done,
                    "obs0_orig": ob,
                    "acs_orig": ac,
                    "obs1_orig": new_ob,
                }
                agent.store_transition(transition)
        else:
            rew = agent.get_syn_rew(ob[None], ac[None], new_ob[None])
            rew = np.asscalar(rew.cpu().numpy().flatten())
            transition = {
                "obs0": ob,
                "acs": ac,
                "obs1": new_ob,
                "rews": rew,
                "dones1": done,
            }
            agent.store_transition(transition)

        # Set current state with the next
        ob = np.array(deepcopy(new_ob))

        if done:
            # Reset agent's noise process
            agent.reset_noise()
            # Reset agent's env
            ob = np.array(env.reset())

        t += 1


def ep_generator(env, agent, render, record):
    """Generator that spits out a trajectory collected during a single episode
    `append` operation is also significantly faster on lists than numpy arrays,
    they will be converted to numpy arrays once complete and ready to be yielded.
    """

    if record:

        def bgr_to_rgb(x):
            _b = np.expand_dims(x[..., 0], -1)
            _g = np.expand_dims(x[..., 1], -1)
            _r = np.expand_dims(x[..., 2], -1)
            rgb_x = np.concatenate([_r, _g, _b], axis=-1)
            del x, _b, _g, _r
            return rgb_x

        kwargs = {'mode': 'rgb_array'}

        def _render():
            return env.render(**kwargs)

    ob = np.array(env.reset())

    if record:
        ob_orig = _render()

    cur_ep_len = 0
    cur_ep_env_ret = 0
    obs = []
    if record:
        obs_render = []
    acs = []
    env_rews = []
    if agent.hps.monitor_mods:
        mods_1 = []
        mods_2 = []
        mods_3 = []

    while True:

        # Predict action
        ac = agent.predict(ob, apply_noise=False)
        # NaN-proof and clip
        ac = np.nan_to_num(ac)
        ac = np.clip(ac, env.action_space.low, env.action_space.high)

        obs.append(ob)
        if record:
            obs_render.append(ob_orig)
        acs.append(ac)
        new_ob, env_rew, done, _ = env.step(ac)

        if agent.hps.monitor_mods:
            if agent.hps.wrap_absorb:
                _ob = np.append(ob, 0)
                _ac = np.append(ac, 0)
                if done and not env._elapsed_steps == env._max_episode_steps:
                    # Wrap with an absorbing state
                    _new_ob = np.append(np.zeros(agent.ob_shape), 1)
                else:
                    _new_ob = np.append(new_ob, 0)
                _, mod_1, mod_2, mod_3 = agent.get_syn_rew(
                    _ob[None], _ac[None], _new_ob[None], monitor_mods=agent.hps.monitor_mods
                )
            else:
                _, mod_1, mod_2, mod_3 = agent.get_syn_rew(
                    ob[None], ac[None], new_ob[None], monitor_mods=agent.hps.monitor_mods
                )

            mod_1 = np.asscalar(mod_1.cpu().numpy().flatten())
            mod_2 = np.asscalar(mod_2.cpu().numpy().flatten())
            mod_3 = np.asscalar(mod_3.cpu().numpy().flatten())

        if render:
            env.render()

        if record:
            ob_orig = _render()

        env_rews.append(env_rew)
        cur_ep_len += 1
        cur_ep_env_ret += env_rew
        ob = np.array(deepcopy(new_ob))

        if agent.hps.monitor_mods:
            mods_1.append(mod_1)
            mods_2.append(mod_2)
            mods_3.append(mod_3)

        if done:
            obs = np.array(obs)
            if record:
                obs_render = np.array(obs_render)
            acs = np.array(acs)
            env_rews = np.array(env_rews)
            out = {"obs": obs,
                   "acs": acs,
                   "env_rews": env_rews,
                   "ep_len": cur_ep_len,
                   "ep_env_ret": cur_ep_env_ret}
            if record:
                out.update({"obs_render": obs_render})

            if agent.hps.monitor_mods:
                mods_1 = np.array(mods_1)
                mods_2 = np.array(mods_2)
                mods_3 = np.array(mods_3)
                out.update({"mods_1": mods_1,
                            "mods_2": mods_2,
                            "mods_3": mods_3})

            yield out

            cur_ep_len = 0
            cur_ep_env_ret = 0
            obs = []
            if record:
                obs_render = []
            acs = []
            env_rews = []
            if agent.hps.monitor_mods:
                mods_1 = []
                mods_2 = []
                mods_3 = []
            ob = np.array(env.reset())

            if record:
                ob_orig = _render()


def evaluate(args,
             env,
             agent_wrapper,
             experiment_name):

    # Create an agent
    agent = agent_wrapper()

    # Create episode generator
    ep_gen = ep_generator(env, agent, args.render)

    if args.record:
        vid_dir = osp.join(args.video_dir, experiment_name)
        os.makedirs(vid_dir, exist_ok=True)

    # Load the model
    agent.load(args.model_path, args.iter_num)
    logger.info("model loaded from path:\n  {}".format(args.model_path))

    # Initialize the history data structures
    ep_lens = []
    ep_env_rets = []
    # Collect trajectories
    for i in range(args.num_trajs):
        logger.info("evaluating [{}/{}]".format(i + 1, args.num_trajs))
        traj = ep_gen.__next__()
        ep_len, ep_env_ret = traj['ep_len'], traj['ep_env_ret']
        # Aggregate to the history data structures
        ep_lens.append(ep_len)
        ep_env_rets.append(ep_env_ret)
        if args.record:
            # Record a video of the episode
            record_video(vid_dir, i, traj['obs_render'])

    # Log some statistics of the collected trajectories
    ep_len_mean = np.mean(ep_lens)
    ep_env_ret_mean = np.mean(ep_env_rets)
    logger.record_tabular("ep_len_mean", ep_len_mean)
    logger.record_tabular("ep_env_ret_mean", ep_env_ret_mean)
    logger.dump_tabular()


def learn(args,
          rank,
          env,
          eval_env,
          agent_wrapper,
          experiment_name):

    # Create an agent
    agent = agent_wrapper()

    # Create context manager that records the time taken by encapsulated ops
    timed = timed_cm_wrapper(logger, use=DEBUG)

    # Start clocks
    num_iters = int(args.num_timesteps) // args.rollout_len
    iters_so_far = 0
    timesteps_so_far = 0
    tstart = time.time()

    if rank == 0:
        # Create collections
        d = defaultdict(list)

        # Set up model save directory
        ckpt_dir = osp.join(args.checkpoint_dir, experiment_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        # Save the model as a dry run, to avoid bad surprises at the end
        agent.save(ckpt_dir, "{}_dryrun".format(iters_so_far))
        logger.info("dry run. Saving model @: {}".format(ckpt_dir))
        if args.record:
            vid_dir = osp.join(args.video_dir, experiment_name)
            os.makedirs(vid_dir, exist_ok=True)

        # Handle timeout signal gracefully
        def timeout(signum, frame):
            # Save the model
            agent.save(ckpt_dir, "{}_timeout".format(iters_so_far))
            # No need to log a message, orterun stopped the trace already
            # No need to end the run by hand, SIGKILL is sent by orterun fast enough after SIGTERM

        # Tie the timeout handler with the termination signal
        # Note, orterun relays SIGTERM and SIGINT to the workers as SIGTERM signals,
        # quickly followed by a SIGKILL signal (Open-MPI impl)
        signal.signal(signal.SIGTERM, timeout)

        # Group by everything except the seed, which is last, hence index -1
        # For 'sam-dac', it groups by uuid + gitSHA + env_id + num_demos,
        # while for 'ddpg-td3', it groups by uuid + gitSHA + env_id
        group = '.'.join(experiment_name.split('.')[:-1])

        # Set up wandb
        while True:
            try:
                wandb.init(
                    project=args.wandb_project,
                    name=experiment_name,
                    id=experiment_name,
                    group=group,
                    config=args.__dict__,
                    dir=args.root,
                )
                break
            except Exception:
                pause = 10
                logger.info("wandb co error. Retrying in {} secs.".format(pause))
                time.sleep(pause)
        logger.info("wandb co established!")

    # Create rollout generator for training the agent
    if args.algo.split('_')[0] == 'ddpg-td3':
        roll_gen = rl_rollout_generator(env, agent, args.rollout_len)
    elif args.algo.split('_')[0] == 'sam-dac':
        roll_gen = il_rollout_generator(env, agent, args.rollout_len)
    else:
        raise NotImplementedError
    if eval_env is not None:
        assert rank == 0, "non-zero rank mpi worker forbidden here"
        # Create episode generator for evaluating the agent
        eval_ep_gen = ep_generator(eval_env, agent, args.render, args.record)

    while iters_so_far <= num_iters:

        if iters_so_far % 100 == 0 or DEBUG:
            log_iter_info(logger, iters_so_far, num_iters, tstart)

        # if iters_so_far % 20 == 0:
        #     # Check if the mpi workers are still synced
        #     sync_check(agent.actr)
        #     sync_check(agent.crit)
        #     if agent.hps.clipped_double:
        #         sync_check(agent.twin)
        #     sync_check(agent.disc)

        with timed("interacting"):
            roll_gen.__next__()  # no need to get the returned rollout, stored in buffer

        if args.algo.split('_')[0] == 'ddpg-td3':

            with timed('training'):
                for training_step in range(args.training_steps_per_iter):

                    if agent.param_noise is not None:
                        if training_step % args.pn_adapt_frequency == 0:
                            # Adapt parameter noise
                            agent.adapt_param_noise()
                        if rank == 0 and iters_so_far % args.eval_frequency == 0:
                            # Store the action-space dist between perturbed and non-perturbed
                            d['pn_dist'].append(agent.pn_dist)
                            # Store the new std resulting from the adaption
                            d['pn_cur_std'].append(agent.param_noise.cur_std)

                    # Sample a batch of transitions from the replay buffer
                    batch = agent.sample_batch()
                    # Update the actor and critic
                    metrics, lrnows = agent.update_actor_critic(
                        batch=batch,
                        update_actor=not bool(iters_so_far % args.actor_update_delay),
                        iters_so_far=iters_so_far,
                    )
                    if rank == 0 and iters_so_far % args.eval_frequency == 0:
                        # Log training stats
                        d['actr_losses'].append(metrics['actr_loss'])
                        d['crit_losses'].append(metrics['crit_loss'])
                        if agent.hps.clipped_double:
                            d['twin_losses'].append(metrics['twin_loss'])
                        if agent.hps.prioritized_replay:
                            iws = metrics['iws']  # last one only

        elif args.algo.split('_')[0] == 'sam-dac':

            with timed('training'):
                for training_step in range(args.training_steps_per_iter):

                    if agent.param_noise is not None:
                        if training_step % args.pn_adapt_frequency == 0:
                            # Adapt parameter noise
                            agent.adapt_param_noise()
                        if rank == 0 and iters_so_far % args.eval_frequency == 0:
                            # Store the action-space dist between perturbed and non-perturbed
                            d['pn_dist'].append(agent.pn_dist)
                            # Store the new std resulting from the adaption
                            d['pn_cur_std'].append(agent.param_noise.cur_std)

                    for _ in range(agent.hps.g_steps):
                        # Sample a batch of transitions from the replay buffer
                        batch = agent.sample_batch()
                        # Update the actor and critic
                        metrics, lrnows = agent.update_actor_critic(
                            batch=batch,
                            update_actor=not bool(iters_so_far % args.actor_update_delay),
                            iters_so_far=iters_so_far,
                        )
                        if rank == 0 and iters_so_far % args.eval_frequency == 0:
                            # Log training stats
                            d['actr_losses'].append(metrics['actr_loss'])
                            d['crit_losses'].append(metrics['crit_loss'])
                            if agent.hps.clipped_double:
                                d['twin_losses'].append(metrics['twin_loss'])
                            if agent.hps.prioritized_replay:
                                iws = metrics['iws']  # last one only

                    for _ in range(agent.hps.d_steps):
                        # Sample a batch of transitions from the replay buffer
                        batch = agent.sample_batch()
                        # Update the discriminator
                        metrics = agent.update_discriminator(batch)
                        if rank == 0 and iters_so_far % args.eval_frequency == 0:
                            # Log training stats
                            d['disc_losses'].append(metrics['disc_loss'])

        if rank == 0 and iters_so_far % args.eval_frequency == 0:

            with timed("evaluating"):

                for eval_step in range(args.eval_steps_per_iter):
                    # Sample an episode w/ non-perturbed actor w/o storing anything
                    eval_ep = eval_ep_gen.__next__()
                    # Aggregate data collected during the evaluation to the buffers
                    d['eval_len'].append(eval_ep['ep_len'])
                    d['eval_env_ret'].append(eval_ep['ep_env_ret'])
                    if agent.hps.monitor_mods:
                        d['mod_1'].extend(eval_ep['mods_1'])
                        d['mod_2'].extend(eval_ep['mods_2'])
                        d['mod_3'].extend(eval_ep['mods_3'])

        # Increment counters
        iters_so_far += 1
        timesteps_so_far += args.rollout_len

        if rank == 0 and ((iters_so_far - 1) % args.eval_frequency == 0):

            # Log stats in csv
            logger.record_tabular('timestep', timesteps_so_far)
            logger.record_tabular('eval_len', np.mean(d['eval_len']))
            logger.record_tabular('eval_env_ret', np.mean(d['eval_env_ret']))
            if agent.hps.monitor_mods:
                logger.record_tabular('mod_1', np.mean(d['mod_1']))
                _mod_2 = np.mean(d['mod_2'])
                _mod_3 = np.mean(d['mod_3'])
                gamma2c = (agent.hps.gamma ** 2) * ((_mod_2 ** 2) * max(1, (_mod_3 ** 2)))
                logger.record_tabular('mod_2', _mod_2)
                logger.record_tabular('mod_3', _mod_3)
                logger.record_tabular('gamma2c', gamma2c)
            logger.info("dumping stats in .csv file")
            logger.dump_tabular()

            if args.record:
                # Record the last episode in a video
                record_video(vid_dir, iters_so_far, eval_ep['obs_render'])

            # Log stats in dashboard
            if agent.hps.prioritized_replay:
                quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
                np.quantile(iws, quantiles)
                wandb.log({"q{}".format(q): np.quantile(iws, q)
                           for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
                          step=timesteps_so_far)
            if agent.param_noise is not None:
                wandb.log({'pn_dist': np.mean(d['pn_dist']),
                           'pn_cur_std': np.mean(d['pn_cur_std'])},
                          step=timesteps_so_far)
            wandb.log({'actr_loss': np.mean(d['actr_losses']),
                       'actr_lrnow': np.array(lrnows['actr']),
                       'crit_loss': np.mean(d['crit_losses'])},
                      step=timesteps_so_far)
            if agent.hps.clipped_double:
                wandb.log({'twin_loss': np.mean(d['twin_losses'])},
                          step=timesteps_so_far)

            if args.algo.split('_')[0] == 'sam-dac':
                wandb.log({'disc_loss': np.mean(d['disc_losses'])},
                          step=timesteps_so_far)

            wandb.log({'eval_len': np.mean(d['eval_len']),
                       'eval_env_ret': np.mean(d['eval_env_ret'])},
                      step=timesteps_so_far)
            if agent.hps.monitor_mods:
                wandb.log({'mod_1': np.mean(d['mod_1']),
                           'mod_2': _mod_2,
                           'mod_3': _mod_3,
                           'gamma2c': gamma2c},
                          step=timesteps_so_far)

            # Clear the iteration's running stats
            d.clear()

    if rank == 0:
        # Save once we are done iterating
        agent.save(ckpt_dir, iters_so_far)
        logger.info("we're done. Saving model @: {}".format(ckpt_dir))
        logger.info("bye.")
