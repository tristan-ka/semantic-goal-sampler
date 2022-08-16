import yaml
import itertools
import argparse


def dict_to_str(dict):
    out_str = ''
    for i, (k, v) in enumerate(dict.items()):
        end_str = '\n' if i == len(dict)-1 else ' '
        out_str += 'rl_script_args.' + k + '=' + str(v) + end_str
    return out_str


def main(args):
    with open(args.input_file, 'r') as json_file:
        variations = yaml.safe_load(json_file)

    keys, values = zip(*variations.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    with open(args.output_file, 'w') as fp:
        for arg_dict in permutations_dicts:
            fp.write(dict_to_str(arg_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    args = parser.parse_args()
    main(args)

    stop = 0
