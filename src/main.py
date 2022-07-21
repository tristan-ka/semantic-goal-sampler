import hydra
import logging
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from playground_env.env_params import get_env_params
from playground_env.descriptions import generate_all_descriptions


def generate_prompt(known_goals):
    prompt = 'You know how to:'
    for goal in known_goals:
        prompt += goal + ', '
    prompt += 'what should you do next?'
    return prompt

def encode_prompt(prompt, model, tokenizer):
    output = tokenizer.encode(prompt, return_tensors="pt")[0]
    # todo: max_seq_len = max_position_embeddings -2 because of sos and eos
    #  this is really ugly, why can't I have access to max_seq_length directly
    # is it an input parameter to the model
    max_len = model.config.encoder.max_position_embeddings -2

    if len(output) > max_len:
        return output[:max_len].reshape([1, max_len])
    else:
        return output.reshape([1, len(output)])

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    model_path = cfg.llm_model_path
    env_params = get_env_params()
    train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)

    prompt = generate_prompt(train_descriptions)
    # model_path = 'model_files/roberta2gpt2-daily-dialog'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    inputs = encode_prompt(prompt, model, tokenizer)
    outputs = model.generate(inputs)

    logging.info(tokenizer.decode(outputs[0]))

if __name__ == '__main__':
    main()