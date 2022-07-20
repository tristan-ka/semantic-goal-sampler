import json
import gym


def compute_babyai_level_instructions(level, n_episodes):
    """
    Randomly sample instructions from a level for n_episodes.
    """
    env = gym.make(level)
    return set(env.reset()['mission'] for i in range(n_episodes))


def write_instructions_to_file(instructions, output_file):
    """
    write instructions to file in json format
    """
    instr_dict = {}
    for i, instr in enumerate(instructions):

        instr_dict[i] = instr

    with open(output_file, 'w') as f:
        json.dump(instr_dict, f)
