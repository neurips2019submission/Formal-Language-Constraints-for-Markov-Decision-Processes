from baselines.constraint.constraint import CountingPotentialConstraint
import numpy as np

def reacher_dynamic_constraint(reward):
    name = "reacher_dynamic"

    state_mean = .2
    state_std = .1
    act_mean = 1.3
    act_std = .65

    def reacher_distance_to_target(obs):
        # print('state norm:', np.linalg.norm(obs[-3:-1]))
        # states.append(np.linalg.norm(obs[-3:-1]))
        # print('STATE mean: ', np.mean(states), 'std:', np.std(states)
        distance = np.linalg.norm(obs[-3:-1])
        discrete = int(distance // 0.1)
        # print('reacher_dynamic_dist_discrete', int(discrete))
        if discrete < 0: discrete = 0
        if discrete > 4: discrete = 4
        return str(discrete)

    def reacher_discretize_action(act):
        # print("act norm:", np.linalg.norm(act))
        # acts.append(np.linalg.norm(act))
        # print('ACT mean: ', np.mean(acts), 'std:', np.std(acts))
        norm = np.linalg.norm(act)
        discrete = int(norm // 0.65)
        # print('reacher_dynamic_act_discrete', int(discrete))
        if discrete < 0: discrete = 0
        if discrete > 4: discrete = 4
        return str(discrete)

    regex = "|".join(['s' + s for s in ["34","23","24","12","13","14","01","02","03","04"]])
    print(regex)

    return CountingPotentialConstraint(
        name, regex, reward, 0.99, s_tl=reacher_distance_to_target, a_tl=reacher_discretize_action)
