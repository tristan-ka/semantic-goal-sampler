import hydra
import logging
import sys
import random
import torch
import re
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.append('../.')
from src.playground_env.env_params import get_env_params
from src.playground_env.descriptions import generate_all_descriptions


def generate_prompt(known_goals, prompt_type='open'):
    prompt = 'Here is a list of goals that you know:'
    r_idx = random.choices([i for i in range(len(known_goals))],k=50)
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


def encode_prompt(prompt, tokenizer):
    output = tokenizer.encode(prompt, return_tensors="pt")[0]
    max_len = 1024

    if len(output) > max_len:
        return output[:max_len].reshape([1, max_len])
    else:
        return output.reshape([1, len(output)])

def prune_output(output_text):
    # TODO: adhoc function based on observed generated text
    # final version should look into env grammar and remove all non environmental tokens
    return re.sub('<.*?>', '', output_text)[3:]

def write_set_to_txt(filename, out_set):
    with open(filename, 'w') as fp:
        for item in out_set:
            fp.write("%s\n" % item)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    device = cfg.device
    logging.info('Device is: ' + device)
    model_path = cfg.llm_model_path
    env_params = get_env_params()
    prompt_type = cfg.prompt_type

    logging.info(prompt_type)
    train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    goal_candidates = []
    for tries in range(cfg.n_gen):
        prompt = generate_prompt(train_descriptions, prompt_type=prompt_type)
        inputs = encode_prompt(prompt, tokenizer).to(device)
        outputs = model.generate(inputs)
        goal_candidate = prune_output(tokenizer.decode(outputs[0]))
        goal_candidates.append(goal_candidate)

    set_candidates = set(goal_candidates)
    set_train = set(train_descriptions)
    set_test = set(test_descriptions)

    write_set_to_txt(r'candidates_U_test.txt', set_candidates.intersection(set_test))
    write_set_to_txt(r'candidates_U_train.txt', set_candidates.intersection(set_train))
    write_set_to_txt(r'others.txt', set_candidates - set_train - set_test)

if __name__ == '__main__':
    main()