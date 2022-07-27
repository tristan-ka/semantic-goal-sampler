import hydra
import logging
import sys
import random
import torch
import re
from math import ceil
from enum import Enum
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForMaskedLM

sys.path.append('../.')
from src.playground_env.env_params import get_env_params
from src.playground_env.descriptions import generate_all_descriptions


class ModelTypesEnum(Enum):
    causal = 0
    seq2seq = 1


def load_hf_model_and_tokenizer(type, path):
    print("Loading model {}".format(path))
    tokenizer = AutoTokenizer.from_pretrained(path)

    # Select class according to type
    if ModelTypesEnum[type] == ModelTypesEnum.causal:
        model = AutoModelForCausalLM.from_pretrained(path)
        n_layers = model.config.n_layer
    elif ModelTypesEnum[type] == ModelTypesEnum.seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(path)
        n_layers = len(model.encoder.block)
    else:
        raise NotImplementedError()

    return tokenizer, model, n_layers


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
    return re.sub('<.*?>', '', output_text)


def write_set_to_txt(filename, out_set):
    with open(filename, 'w') as fp:
        for item in out_set:
            fp.write("%s\n" % item)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    model_path = cfg.llm_model_path
    devices = cfg.devices

    if 'distilgpt2' in model_path:
        llm_type = 'causal'
    elif 't0pp' in model_path:
        llm_type = 'seq2seq'
    else:
        raise ValueError('Unknown LLM type')

    tokenizer, model, num_layers = load_hf_model_and_tokenizer(llm_type, model_path)
    device = devices[0]
    if len(devices) == 1:
        model.to(device)
    else:
        layers_per_device = ceil(num_layers / len(devices))
        device_map = {
            _device: list(range(i * layers_per_device, min((i + 1) * layers_per_device, num_layers)))
            for i, _device in enumerate(devices)
        }
        model.parallelize(device_map)

    logging.info('Device is: ' + device)

    env_params = get_env_params()
    prompt_type = cfg.prompt_type
    n_goals_prompt = cfg.n_goals

    logging.info(prompt_type)
    train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)


    goal_candidates = []
    for tries in range(cfg.n_gen):
        prompt = generate_prompt(known_goals=train_descriptions,
                                 prompt_type=prompt_type,
                                 n_goals=n_goals_prompt)
        inputs = encode_prompt(prompt, tokenizer).to(device)
        outputs = model.generate(inputs)
        goal_candidate = prune_output(tokenizer.decode(outputs[0]))
        goal_candidates.append(goal_candidate)

    set_candidates = set(goal_candidates)
    write_set_to_txt(r'others.txt', set_candidates)


if __name__ == '__main__':
    main()
