import hydra
import logging
import sys
import random
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.append('../.')
from src.playground_env.env_params import get_env_params
from src.playground_env.descriptions import generate_all_descriptions


def generate_prompt(known_goals):
    prompt = 'Here is a list of goals that you know:'
    r_idx = random.choices([i for i in range(len(known_goals))],k=40)
    for goal in [known_goals[i] for i in r_idx]:
        prompt += goal + ', '
    prompt += 'what goals could you imagine from this list that is not in the list?'
    return prompt

def encode_prompt(prompt, model, tokenizer):
    output = tokenizer.encode(prompt, return_tensors="pt")[0]
    # todo: max_seq_len = max_position_embeddings -2 because of sos and eos
    #  this is really ugly, why can't I have access to max_seq_length directly
    # is it an input parameter to the model
    max_len = 1024#model.config.encoder.max_position_embeddings -2

    if len(output) > max_len:
        return output[:max_len].reshape([1, max_len])
    else:
        return output.reshape([1, len(output)])

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    device = "cpu"#"cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info('Device is: ' + device)
    model_path = cfg.llm_model_path
    env_params = get_env_params()
    train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)

    
    # model_path = 'model_files/roberta2gpt2-daily-dialog'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    output_text = []
    for tries in range(100):
        prompt = generate_prompt(train_descriptions)
        inputs = encode_prompt(prompt, model, tokenizer).to(device)
        outputs = model.generate(inputs)
        output_text.append(tokenizer.decode(outputs[0]))

    logging.info(tokenizer.decode(outputs[0]))
    with open(r'output_goals.txt', 'w') as fp:
        for item in output_text:
            # write each item on a new line
            fp.write("%s\n" % item)
            logging.info(item)
        

if __name__ == '__main__':
    main()