import argparse
from utils import compute_babyai_level_instructions, write_instructions_to_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Show level instructions")
    parser.add_argument("--n_episodes", type=int, default=10000,
                        help="Collect instructions from this many episodes")
    parser.add_argument("--level",
                        help="The level of interest")
    args = parser.parse_args()
    instructions = compute_babyai_level_instructions(args.level, args.n_episodes)
    write_instructions_to_file(instructions, args.level + '_instructions.json')
