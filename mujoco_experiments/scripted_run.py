#!venv/bin/python
import signal
import subprocess
import argparse
import os

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'env',
    type=str,
    choices=[
        'reacher_actuation_baseline', 'reacher_dynamic_baseline', 'cheetah_dithering_baseline',
        'reacher_actuation', 'reacher_dynamic', 'cheetah_dithering'
    ])
parser.add_argument('augmentation', type=str)
parser.add_argument('reward_mod', type=int)
parser.add_argument('reward_mod_str', type=str)
parser.add_argument('base_dir_num', type=int)
args = parser.parse_args()


def signal_handler(signal, frame):
    global interrupted
    interrupted = True


signal.signal(signal.SIGINT, signal_handler)

interrupted = False
i = args.base_dir_num
while True:
    print("Starting another run!")
    my_env = os.environ.copy()
    if args.env == 'reacher_dynamic':
        my_env["OPENAI_LOGDIR"] = "reacher_dynamic_{}_{}".format(
            args.reward_mod_str, i)
        subprocess.call(
            'python -m baselines.run --constraints reacher_dynamic --rewards {} --augmentation {}'
            .format(args.reward_mod, args.augmentation).split(),
            env=my_env)
    if args.env == 'reacher_actuation':
        my_env["OPENAI_LOGDIR"] = "reacher_actuation_{}_{}".format(
            args.reward_mod_str, i)
        subprocess.call(
            'python -m baselines.run --constraints reacher_actuation_counting --rewards {} --augmentation {}'
            .format(args.reward_mod, args.augmentation).split(),
            env=my_env)
    elif args.env == 'cheetah_dithering':
        my_env["OPENAI_LOGDIR"] = "cheetah_dithering_{}_{}".format(
            args.reward_mod_str, i)
        subprocess.call(
            'python -m baselines.run --augmentation {aug} --env HalfCheetah-v2 --constraints half_cheetah_dithering_0 half_cheetah_dithering_1 half_cheetah_dithering_2 half_cheetah_dithering_3 half_cheetah_dithering_4 half_cheetah_dithering_5 --rewards {r} {r} {r} {r} {r} {r}'
            .format(aug=args.augmentation, r=args.reward_mod).split(),
            env=my_env)

    print("All done!")
    i += 1

    if interrupted:
        print("Gotta go")
        break
