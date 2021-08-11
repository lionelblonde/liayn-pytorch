import argparse

from helpers.misc_util import boolean_flag


def argparser(description="DDPG Experiment"):
    """Create an argparse.ArgumentParser"""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--wandb_project', help='wandb project name', default='DEFAULT')
    parser.add_argument('--env_id', help='environment identifier', default=None)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', help='demos location', type=str, default=None)
    parser.add_argument('--uuid', type=str, default=None)
    boolean_flag(parser, 'cuda', default=False)
    boolean_flag(parser, 'render', help='render the interaction traces', default=False)
    boolean_flag(parser, 'record', help='record the interaction traces', default=False)
    parser.add_argument('--task', type=str, choices=['train', 'eval'], default=None)
    parser.add_argument('--algo', type=str, choices=['ddpg-td3', 'sam-dac'], default=None)

    # Training
    parser.add_argument('--num_timesteps', help='total number of interactions',
                        type=int, default=int(1e7))
    parser.add_argument('--training_steps_per_iter', type=int, default=4)
    parser.add_argument('--eval_steps_per_iter', type=int, default=10)
    parser.add_argument('--eval_frequency', type=int, default=10)

    # Model
    boolean_flag(parser, 'layer_norm', default=False)

    # Optimization
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-4)
    parser.add_argument('--lr_schedule', type=str, choices=['constant', 'linear'], default='constant')
    parser.add_argument('--clip_norm', type=float, default=1.)
    parser.add_argument('--wd_scale', help='weight decay scale', type=float, default=0.001)

    # Algorithm
    parser.add_argument('--rollout_len', help='number of interactions per iteration',
                        type=int, default=2)
    parser.add_argument('--batch_size', help='minibatch size', type=int, default=128)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.99)
    parser.add_argument('--mem_size', type=int, default=int(1e5))
    parser.add_argument('--noise_type', type=str, default='adaptive-param_0.2, ou_0.1, normal_0.1')
    parser.add_argument('--pn_adapt_frequency', type=float, default=50)
    parser.add_argument('--polyak', type=float, default=0.005, help='soft target nets update')
    parser.add_argument('--targ_up_freq', type=int, default=100, help='hard target nets update')
    boolean_flag(parser, 'n_step_returns', default=True)
    parser.add_argument('--lookahead', help='num lookahead steps', type=int, default=10)
    boolean_flag(parser, 'ret_norm', default=False)
    boolean_flag(parser, 'popart', default=False)

    # TD3
    boolean_flag(parser, 'clipped_double', default=False)
    boolean_flag(parser, 'targ_actor_smoothing', default=False)
    parser.add_argument('--td3_std', type=float, default=0.2,
                        help='std of smoothing noise applied to the target action')
    parser.add_argument('--td3_c', type=float, default=0.5,
                        help='limit for absolute value of target action smoothing noise')
    parser.add_argument('--actor_update_delay', type=int, default=1,
                        help='number of critic updates to perform per actor update')

    # Prioritized replay
    boolean_flag(parser, 'prioritized_replay', default=False)
    parser.add_argument('--alpha', help='how much prioritized', type=float, default=0.3)
    parser.add_argument('--beta', help='importance weights usage', type=float, default=1.0)
    boolean_flag(parser, 'ranked', default=False)
    boolean_flag(parser, 'unreal', default=False)

    # Distributional RL
    boolean_flag(parser, 'use_c51', default=False)
    boolean_flag(parser, 'use_qr', default=False)
    parser.add_argument('--c51_num_atoms', type=int, default=51)
    parser.add_argument('--c51_vmin', type=float, default=0.)
    parser.add_argument('--c51_vmax', type=float, default=100.)
    parser.add_argument('--num_tau', type=int, default=200)

    # Adversarial imitation
    parser.add_argument('--g_steps', type=int, default=3)
    parser.add_argument('--d_steps', type=int, default=1)
    parser.add_argument('--d_lr', type=float, default=1e-5)
    boolean_flag(parser, 'state_only', default=True)
    boolean_flag(parser, 'minimax_only', default=True)
    parser.add_argument('--ent_reg_scale', type=float, default=0.001)
    parser.add_argument('--num_demos', help='number of expert demo trajs for imitation',
                        type=int, default=None)
    boolean_flag(parser, 'spectral_norm', help='whether to use spectral norm', default=True)
    boolean_flag(parser, 'grad_pen', help='whether to use gradient penalty', default=True)
    parser.add_argument('--grad_pen_targ', type=float, default=1.)
    parser.add_argument('--grad_pen_scale', type=float, default=10.)
    parser.add_argument('--grad_pen_type', type=str, default=None,
                        choices=['wgan', 'dragan', 'nagard'])
    boolean_flag(parser, 'one_sided_pen', help='whether to use the one-sided version', default=True)
    boolean_flag(parser, 'historical_patching', default=True)
    parser.add_argument('--fake_ls_type', type=str, default='none')
    parser.add_argument('--real_ls_type', type=str, default='random-uniform_0.7_1.2')
    boolean_flag(parser, 'wrap_absorb', default=False)
    boolean_flag(parser, 'd_batch_norm', default=False)

    parser.add_argument('--reward_type', type=str, default=None, choices=['red', 'gail', 'gail_mod_f'])
    parser.add_argument('--f_grad_pen_targ', type=float, default=9.0)
    boolean_flag(parser, 'monitor_mods', default=False)

    parser.add_argument('--red_epochs', type=int, default=200)
    boolean_flag(parser, 'red_batch_norm', default=True)
    parser.add_argument('--red_lr', type=float, default=5e-4)
    parser.add_argument('--proportion_of_exp_per_red_update', type=float, default=1.)

    boolean_flag(parser, 'use_purl', default=False)
    parser.add_argument('--purl_eta', type=float, default=0.25)

    # Evaluation
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--num_trajs', help='number of trajectories to evaluate',
                        type=int, default=10)
    parser.add_argument('--iter_num', help='iteration to evaluate the model at',
                        type=str, default=None)  # the number might have a suffix

    return parser
