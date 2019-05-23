import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')  # Can change to 'Agg' for non-interactive mode

EPISODES_WINDOW = 100
COLORS = [
    'blue', 'red', 'green', 'magenta', 'black', 'purple',
    'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime',
    'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold',
    'lightpurple', 'darkred', 'darkblue'
]


def accumulate_episodes(done, a):
    episode_stops = np.nonzero(done)[0]
    episode_vals = np.zeros((len(episode_stops), ) + a.shape[1:])
    for i, e in enumerate(episode_stops):
        if i == 0:
            episode_vals[0] = np.sum(a[:e])
        else:
            episode_vals[i] = np.sum(a[episode_stops[i - 1]:e])
    return episode_vals


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def window_func(x, y, window, func):
    yw = rolling_window(y, window)
    yw_func = func(yw, axis=-1)
    return x[window - 1:], yw_func


def plot_curves(xy_list, xaxis, yaxis, title, save_path, plot_mean=True, variance=[]):
    fig = plt.figure(figsize=(8, 2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i]
        if len(variance) > i: plt.fill_between(x, y-variance[i], y+variance[i], color=color, alpha=0.35)
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i]
        plt.scatter(x, y, s=2, color=color)
        x, y_mean = window_func(
            x, y, EPISODES_WINDOW,
            np.mean)  #So returns average of last EPISODE_WINDOW episodes
        if plot_mean: plt.plot(x, y_mean, color=color)
    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout()
    fig.canvas.mpl_connect('resize_event', lambda event: plt.tight_layout())
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    return y_mean


def plot_and_window(sequence, sequence_name, done, task_name, directory):
    plot_curves([(np.arange(len(sequence)), sequence)], 'timestep',
                sequence_name, task_name + ' ' + sequence_name,
                os.path.join(directory, task_name + '_step_{}'.format(sequence_name)))
    episode_sequence = accumulate_episodes(done, sequence)
    windowed_episode_sequence = plot_curves(
        [(np.arange(len(episode_sequence)), episode_sequence)], 'episode',
        sequence_name, task_name + '{} per episode'.format(sequence_name),
        os.path.join(directory,
                     task_name + '_episode_step_{}'.format(sequence_name)))
    return windowed_episode_sequence


def select_best_index(key, dictionary):
    idx = np.argmax(dictionary[key])
    return {k: v[idx] for k, v in dictionary.items()}


def truncate_match_sequences(sequences):
    min_len = min(map(len, sequences))
    return [s[:min_len] for s in sequences]


def process_dir(dir):
    if os.path.exists(os.path.join(dir, 'result.json')): return

    # load info about constraints
    with open(os.path.join(dir, 'args.json')) as args_file:
        task_args = json.load(args_file)
        constraints = task_args[
            'constraints'] if 'constraints' in task_args.keys() else []
        violation_values = task_args[
            'rewards'] if 'rewards' in task_args.keys() else []
        env = task_args['env']
        task_name = env

    # load and calculate on rewards
    done = np.load(dir + '/done.npy')
    rewards = np.load(dir + '/reward.npy')
    mean_episode_rewards = plot_and_window(rewards, 'reward', done, task_name, dir)

    # calculate info on constraint violations
    reward_mods_dict = {}
    mean_episode_violations_dict = {}

    for constraint in constraints:
        violations = np.load(dir + '/' + constraint + '_viols.npy')
        reward_mods_dict[constraint] = np.load(dir + '/' + constraint +
                                          '_rew_mod.npy')
        mean_episode_violations_dict[constraint] = plot_and_window(
            violations, constraint, done, task_name, dir)

    # calculate raw reward
    # truncate all reward squences to be the same length for binary ops
    truncated = truncate_match_sequences([rewards] + list(reward_mods_dict.values()))
    rewards, reward_mods = truncated[0], truncated[1:]

    total_reward_mod = sum(reward_mods)
    raw_rewards = rewards - total_reward_mod
    mean_raw_rewards = plot_and_window(raw_rewards, 'raw_reward', done,
                                       task_name, dir)

    # find best reward/violations set
    best_mean_episode_violations_dict = select_best_index(
        'reward', {**{'reward': mean_raw_rewards}, **mean_episode_violations_dict})
    best_mean_raw_reward = best_mean_episode_violations_dict.pop('reward')
    print('{} with {} episode val and constraint {} with shaping val {}'.format(task_name, best_mean_raw_reward, constraint, violation_values[-1]))

    best_mean_vals_dict = {
        'raw_reward': best_mean_raw_reward,
        'violations': best_mean_episode_violations_dict
    }
    with open(os.path.join(dir, 'result.json'), 'w') as result_file:
        json.dump(best_mean_vals_dict, result_file)

    if not constraints:
        print(task_name)
        print(select_best_index('reward', {'reward', mean_episode_rewards}))

    return best_mean_vals_dict, mean_raw_rewards, mean_episode_violations_dict

def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'dirs', help='List of log directories', nargs='*', default=['./log'])
    parser.add_argument('--agg', action='store_true')
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--cross_cut_reward', nargs="*", default=[])
    parser.add_argument('--cross_cut_constraint', nargs="*", default=[])
    args = parser.parse_args()
    args.dirs = [os.path.abspath(dir) for dir in args.dirs]
    best_mean_vals_dicts_list = []
    raw_rewards_list = []
    mean_episode_violations_dict_list = []
    if args.cross_cut_reward:
        vals = []
        std_devs = []
        for tn in args.cross_cut_plot:
            vals.append(np.load('{}_mean_raw_rewards_per_episode.npy'.format(tn)))
            std_devs.append(np.load('{}_stddev_raw_rewards_per_episode.npy'.format(tn)))
        vals = truncate_match_sequences(vals)
        std_devs = truncate_match_sequences(std_devs)
        plot_curves([(np.arange(len(v)), v) for v in vals], 'episode',
                'aggregate raw rewards', args.task_name + ' ' + 'aggregate raw rewards',
                args.task_name + '_aggregate_raw_rewards', plot_mean=False, variance=std_devs)
        exit()
    if args.cross_cut_constraint:
        vals = []
        std_devs = []
        key = args.cross_cut_constraint[0]
        for tn in args.cross_cut_constraint[1:]:
            vals.append(np.load('{}_constraint_{}_mean_violations_per_episode.npy'.format(tn, key)))
            std_devs.append(np.load('{}_constraint_{}_stddev_violations_per_episode.npy'.format(tn, key)))
        vals = truncate_match_sequences(vals)
        std_devs = truncate_match_sequences(std_devs)
        plot_curves([(np.arange(len(v)), v) for v in vals], 'episode',
                'constraint {}'.format(key), args.task_name + ' constraint {}'.format(key),
                args.task_name + '_constraint_{}'.format(key), plot_mean=False, variance=std_devs)
        exit()
    
    for directory in args.dirs:
        best_mean_vals_dict, raw_rewards, mean_episode_violations_dict = process_dir(directory)
        best_mean_vals_dicts_list.append(best_mean_vals_dict)
        raw_rewards_list.append(raw_rewards)
        mean_episode_violations_dict_list.append(mean_episode_violations_dict)
    if args.agg:
        if not os.path.exists(args.task_name):
            os.makedirs(args.task_name)
        raw_rewards_list = truncate_match_sequences(raw_rewards_list)
        agg_raw_rewards = np.mean(raw_rewards_list, axis=0)
        std_raw_rewards = np.std(raw_rewards_list, axis=0)
        plot_curves([(np.arange(len(agg_raw_rewards)), agg_raw_rewards)], 'episode',
                'aggregate raw rewards', args.task_name + ' ' + 'aggregate raw rewards',
                os.path.join(args.task_name, args.task_name + '_aggregate_raw_rewards'), plot_mean=False, variance=[std_raw_rewards])
        np.save('{}_mean_raw_rewards_per_episode'.format(args.task_name), agg_raw_rewards)
        np.save('{}_stddev_raw_rewards_per_episode'.format(args.task_name), std_raw_rewards)

        for key in mean_episode_violations_dict_list[0].keys():
            relevant_vals = [d[key] for d in mean_episode_violations_dict_list]
            relevant_vals = truncate_match_sequences(relevant_vals)
            agg_relevant_vals = np.mean(relevant_vals, axis=0)
            std_relevant_vals = np.std(relevant_vals, axis=0)
            plot_curves([(np.arange(len(agg_relevant_vals)), agg_relevant_vals)], 'episode',
                    'constraint {}'.format(key), args.task_name + ' constraint {}'.format(key),
                    os.path.join(args.task_name, args.task_name + '_constraint_{}'.format(key)), plot_mean=False, variance=[std_relevant_vals])
            np.save('{}_constraint_{}_mean_violations_per_episode'.format(args.task_name, key), agg_relevant_vals)
            np.save('{}_constraint_{}_stddev_violations_per_episode'.format(args.task_name, key), std_relevant_vals)


if __name__ == '__main__':
    main()