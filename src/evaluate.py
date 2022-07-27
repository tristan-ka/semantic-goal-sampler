import re
import yaml
from src.playground_env.env_params import get_env_params
from src.playground_env.descriptions import generate_all_descriptions

if __name__ == '__main__':

    env_params = get_env_params()
    train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)

    output_dir = '/Users/tristankarch/Repo/semantic-goal-sampler/src/outputs/jeanzay/17-30-50/'
    with open(output_dir + 'others.txt') as f:
        lines = f.readlines()
    with open(output_dir + '.hydra/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    goals = []
    for line in lines:
        goal = line[1:-1]
        if ',' in goal:
            goals.extend(goal.split(', '))
        else:
            goals.append(goal)

    goal_candidates = set(goals)
    train_goals = set(train_descriptions)
    test_goals = set(test_descriptions)
    other_goals = goal_candidates - train_goals - test_goals

    G_U_test = goal_candidates.intersection(test_goals)
    G_U_train = goal_candidates.intersection(train_goals)
    G_out = goal_candidates - train_goals - test_goals
    stop = 0
