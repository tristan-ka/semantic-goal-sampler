import hydra
import logging
import sys
import random
import re
from enum import Enum
from omegaconf import DictConfig
from lamorel import Caller
from accelerate import Accelerator
from hydra.utils import to_absolute_path

accelerator = Accelerator()

sys.path.append('../.')
sys.path.append('src/')
from playground_env.env_params import get_env_params
from playground_env.descriptions import generate_all_descriptions


def generate_prompt(known_goals, prompt_type='open', n_goals=50):
    prompt = 'Here is a list of goals that you know:'
    r_idx = random.choices([i for i in range(len(known_goals))], k=n_goals)
    for goal in [known_goals[i] for i in r_idx]:
        prompt += goal + ', '
    if prompt_type == 'open':
        prompt += 'what goals could you imagine from this list that is not in the list?'
    elif prompt_type == 'predicate':
        predicates = ['grow', 'grasp']
        prompt += 'what else can you ' + random.choice(predicates) + '?'
    else:
        raise ValueError('Please provide a valid prompt_type')
    return prompt


def prune_output(output_texts):
    # TODO: adhoc function based on observed generated text
    # final version should look into env grammar and remove all non environmental tokens

    return [re.sub('<.*?>', '', text) for text in output_texts]


def write_set_to_txt(filename, out_set):
    with open(filename, 'w') as fp:
        for item in out_set:
            fp.write("%s\n" % item)


@hydra.main(config_path="../conf", config_name="local_config", )
def main(cfg: DictConfig) -> None:
    lm_server = Caller(cfg.lamorel_args)

    cfg_rl = cfg.rl_script_args
    env_params = get_env_params()
    prompt_type = cfg_rl.prompt_type
    n_goals_prompt = cfg_rl.n_goals

    logging.info(prompt_type)
    train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)

    if cfg_rl.generation_type == 'prob':
        generation_kwargs = {
            'max_new_tokens': 15,
            'do_sample': True,
            'temperature': 1,
            'top_p': 0.7,
            'top_k': 0,
            'max_length': 128
        }
    elif cfg_rl.generation_type == 'deter':
        generation_kwargs = {
            'max_length': 128
        }
    else:
        raise ValueError(
            "{} unkown. Please enter a valid generation type (prob) or (deter)".format(cfg_rl.generation_type))

    output_goals = []
    for tries in range(cfg_rl.n_gen):
        print(tries)
        prompts = [generate_prompt(known_goals=train_descriptions,
                                   prompt_type=prompt_type,
                                   n_goals=n_goals_prompt) for _ in range(cfg_rl.batch_size_gen)]
        outputs = lm_server.generate(contexts=prompts, **generation_kwargs)

        outputs_text = [output[0]['text'] for output in outputs]
        output_goals.extend(outputs_text)

    set_goals = set(output_goals)
    set_pruned_goals = set(prune_output(output_goals))
    write_set_to_txt(r'goals.txt', set_goals)
    write_set_to_txt(r'pruned_goals.txt', set_pruned_goals)
    lm_server.close()


if __name__ == '__main__':
    main()
