import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import variations2permutations
from playground_env.env_params import get_env_params
from playground_env.descriptions import generate_all_descriptions

if __name__ == '__main__':

    env_params = get_env_params()
    train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)

    variations_file = '/Users/tristankarch/Repo/semantic-goal-sampler/conf/variations/variations.yaml'
    with open(variations_file) as fp:
        variation_dict = yaml.load(fp, Loader=yaml.FullLoader)

    permutations_dicts = variations2permutations(variation_dict)

    df_result = pd.DataFrame()
    for v_dict in permutations_dicts:
        key_dir = '_'.join([k + ':' + str(v) for k, v in v_dict.items()])
        output_root = '/Users/tristankarch/Repo/semantic-goal-sampler/outputs/outputs_jeanzay'
        output_dir = os.path.join(output_root, key_dir, 'outputs', key_dir)

        with open(os.path.join(output_dir, 'pruned_goals.txt')) as f:
            lines = f.readlines()

        with open(os.path.join(output_dir, '.hydra/config.yaml')) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        goals = []
        for line in lines:
            goal = line[:-1]
            if ',' in goal:
                goals.extend(goal.split(', '))
            else:
                goals.append(goal)

        goal_candidates = set(goals)
        train_goals = set(train_descriptions)
        test_goals = set(test_descriptions)
        other_goals = goal_candidates - train_goals - test_goals

        result_dict = {}
        result_dict['G_U_test'] = [goal_candidates.intersection(test_goals)]
        result_dict['G_U_train'] = [goal_candidates.intersection(train_goals)]
        result_dict['G_out'] = [goal_candidates - train_goals - test_goals]
        result_dict['test_coverage'] = len(goal_candidates.intersection(test_goals)) / len(test_goals)
        result_dict['precision'] = len(goal_candidates.intersection(test_goals)) / (
                len(goal_candidates - train_goals))
        result_dict['other_size'] = len(goal_candidates - train_goals - test_goals)
        result_dict['key_dir'] = '\n'.join([k + ':' + str(v) for k, v in v_dict.items()])

        df_result = df_result.append(pd.DataFrame(result_dict), ignore_index=True)

    # Adding Imagine GCH results:
    # result_dict['G_U_test'] = [[]]
    # result_dict['G_U_train'] = [[]]
    # result_dict['G_out'] = [[]]
    # result_dict['test_coverage'] = 0.87
    # result_dict['other_size'] = 72
    # result_dict['precision'] = 0.41
    # result_dict['key_dir'] = 'Imagine'
    # df_result = df_result.append(pd.DataFrame(result_dict), ignore_index=True)

    plt.style.use('ggplot')


    def on_resize(event):
        fig = plt.gcf()
        fig.tight_layout()
        fig.canvas.draw()


    fig1 = plt.figure()
    ax = sns.barplot(x='key_dir', y='test_coverage', data=df_result, capsize=.025)
    plt.title('Coverage')
    plt.ylabel('Coverage')
    plt.ylim([-0.01, 1.11])
    plt.xticks(rotation=30, fontsize=8)
    plt.axhline(y=0.87, xmin=0, xmax=3.5, c='#534B62')
    xlims = ax.get_xlim()
    text_pos = 0.75 * xlims[1]
    plt.text(x=text_pos, y=0.9, s='Imagine Construction Grammar', c='#534B62')
    plt.tight_layout()
    cid1 = fig1.canvas.mpl_connect('resize_event', on_resize)

    fig2 = plt.figure()
    ax = sns.barplot(x='key_dir', y='precision', data=df_result, capsize=.025)
    plt.title('Precision')
    plt.ylabel('Precision')
    plt.ylim([-0.01, 1.11])
    plt.xticks(rotation=30, fontsize=8)
    plt.axhline(y=0.41, xmin=0, xmax=3.5, c='#534B62')
    plt.text(x=text_pos, y=0.44, s='Imagine Construction Grammar', c='#534B62')
    plt.tight_layout()

    # plt.figure()
    # ax = sns.barplot(x='key_dir', y='other_size', data=df_result, capsize=.025)
    # plt.title('Size of Other goals')
    # plt.ylabel('Size of Other goals')
    # # plt.ylim([-0.01, 1.11])
    # plt.xticks(rotation=30, fontsize=8)
    # plt.tight_layout()

    cid2 = fig2.canvas.mpl_connect('resize_event', on_resize)
    plt.show()

    stop = 0
