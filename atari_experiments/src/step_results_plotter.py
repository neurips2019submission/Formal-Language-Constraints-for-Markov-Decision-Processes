import numpy as np
import matplotlib
matplotlib.use('Agg')  # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt
import json
import os

EPISODES_WINDOW = 100
COLORS = [
    'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple',
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


def plot_curves(xy_list, xaxis, yaxis, title, save_path):
    fig = plt.figure(figsize=(8, 2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        color = COLORS[i]
        plt.scatter(x, y, s=2)
        x, y_mean = window_func(
            x, y, EPISODES_WINDOW,
            np.mean)  #So returns average of last EPISODE_WINDOW episodes
        plt.plot(x, y_mean, color=color)
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


def best_reward_violations(rewards, violations):
    idx = np.argmax(rewards)
    return rewards[idx], violations[idx]


def process_dir(dir):
    if os.path.exists(os.path.join(dir, 'result.json')): return
        
    done = np.load(dir + '/done.npy')
    rewards = np.load(dir + '/reward.npy')

    with open(os.path.join(dir, 'args.json')) as args_file:
        task_args = json.load(args_file)
        constraint = task_args['contract']
        violation_val = task_args['violation_reward']
        env = task_args['env_name']
        task_name = env

    episode_rewards = accumulate_episodes(done, rewards)
    mean_ep_rewards = plot_curves(
        [(np.arange(len(episode_rewards)), episode_rewards)], 'episode',
        'reward', task_name + ' episode rewards',
        os.path.join(dir, task_name + '_episode_step_reward'))

    plot_curves([(np.arange(len(rewards)), rewards)], 'timestep', 'reward',
                task_name + ' rewards',
                os.path.join(dir, task_name + '_step_reward'))

    violations = np.load(dir + '/violations.npy')

    reward_mods = np.load(dir + '/reward_mod.npy')
    steps = min((len(rewards), len(reward_mods)))
    if len(rewards) != len(reward_mods):
        rewards = rewards[:steps]
        reward_mods = reward_mods[:steps]
    
    raw_rewards = rewards - reward_mods
    plot_curves([(np.arange(len(violations)), violations)], 'timestep',
                'violation', task_name + ' violations',
                os.path.join(dir, task_name + '_step_violation'))
    plot_curves([(np.arange(len(raw_rewards)), raw_rewards)], 'timestep',
                'raw_reward', task_name + ' raw rewards',
                os.path.join(dir, task_name + '_step_rawreward'))

    episode_violations = accumulate_episodes(done, violations)
    episode_raw_rewards = accumulate_episodes(done, raw_rewards)
    mean_ep_violations = plot_curves(
        [(np.arange(len(episode_violations)), episode_violations)],
        'episode', 'violation', task_name + ' episode violations',
        os.path.join(dir, task_name + '_episode_step_violation'))
    mean_raw_rewards = plot_curves(
        [(np.arange(len(episode_raw_rewards)), episode_raw_rewards)],
        'episode', 'raw_reward', task_name + ' episode raw rewards',
        os.path.join(dir, task_name + '_episode_step_rawreward'))
    print(task_name)
    print('{} with {} val @ {} steps'.format(constraint, violation_val, steps))

    rr, viols = best_reward_violations(mean_raw_rewards, mean_ep_violations)
    best_mean_vals = {'raw_reward': rr, 'violations': viols}
    print(best_mean_vals)
    with open(os.path.join(dir, '_result.json'), 'w') as result_file:
        json.dump(best_mean_vals, result_file)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dirs', help='List of log directories', nargs='*', default=['./log'])
    args = parser.parse_args()
    args.dirs = [os.path.abspath(dir) for dir in args.dirs]
    for dir in args.dirs:
        process_dir(dir)


if __name__ == '__main__':
    main()
