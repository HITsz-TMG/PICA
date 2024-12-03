import logging
import random
import numpy as np
import torch
import os
import transformers
from peft import PeftModel
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, MistralForCausalLM
from typing import List


def set_seed(seed: int) -> None:
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments

    Parameters:
    seed: the number to set the seed to

    Return: None
    """

    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def intervention_mode2list(mode, layers, prefix=""):
    if mode == 'Wo':
        return [f'{prefix}model.layers.{i}.self_attn.o_proj' for i in range(layers)] + \
               [f"{prefix}model.embed_tokens"]
    else:
        raise KeyError


def sv_format(
        tokenizer,
        query: str,
        answer: str = None,
        demon_list: List = None,
        system: str = '',
        proj_tokens: str = '→',
        eos: str = None,
        query_format:str=None,
        demon_proj=None,
):
    if demon_list is None: demon_list = []
    if eos is None:  eos = ''
    if system is None: system = ''
    if query_format is None: query_format = '{}'
    if demon_proj is None: demon_proj = proj_tokens

    SEP_TOKENS = [' ', '\n', '\t']

    def tokenize(target_str, add_sep=True):
        nonlocal sentence
        if target_str is None or target_str == '': return []
        if len(sentence) == 0 or sentence[-1] in SEP_TOKENS or target_str[0] in SEP_TOKENS or not add_sep:
            target_str = target_str
        else:
            target_str = " " + target_str
        source = tokenizer([sentence], truncation=False, padding=False, add_special_tokens=False, return_tensors="pt").input_ids
        target = tokenizer([sentence + target_str], truncation=False, padding=False, add_special_tokens=False, return_tensors="pt").input_ids
        assert (target[0, :len(source[0])] != source[0]).sum() == 0, f"sentence: {sentence}, target_str: {target_str}"
        sentence += target_str
        return target[0, len(source[0]):].tolist()

    sentence = system
    input_tokens = tokenizer(sentence, truncation=False, padding=False, add_special_tokens=True).input_ids
    input_mask = ['bos'] + ['system'] * (len(input_tokens) - 1)

    for r, (q, a) in enumerate(demon_list):
        q = query_format.format(q.strip(' '))
        input_tokens += tokenize(q)
        input_mask += [f'query_{r}'] * (len(input_tokens) - len(input_mask))
        input_tokens += tokenize(demon_proj)
        input_mask += [f'project_{r}'] * (len(input_tokens) - len(input_mask))
        input_tokens += tokenize(a.strip(' '))
        input_mask += [f'answer_{r}'] * (len(input_tokens) - len(input_mask))
        input_tokens += tokenize(eos, add_sep=False)
        input_mask += [f'eos_{r}'] * (len(input_tokens) - len(input_mask))

    if query is not None:
        query = query_format.format(query.strip(' '))
        input_tokens += tokenize(query)
        input_mask += [f'query'] * (len(input_tokens) - len(input_mask))
        input_tokens += tokenize(proj_tokens)
        input_mask += [f'project'] * (len(input_tokens) - len(input_mask))

    if answer:
        input_tokens += tokenize(answer.strip(' '))
        input_mask += [f'answer'] * (len(input_tokens) - len(input_mask))

    input_ids = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)

    return input_ids, input_mask


def sv_format_length(
        tokenizer,
        query:str,
        answer:str=None,
        demon_list:List=None,
        system:str='',
        proj_tokens: str = '→',
        eos: str = None,
        query_format:str=None,
        max_len:int=None,
        reverse=True,
        demon_proj=None,
):
    if demon_list is None:
        demon_list = []
    if max_len is None:
        max_len = tokenizer.model_max_length
    if reverse:
        for i in range(len(demon_list) + 1):
            input_ids, input_mask = sv_format(tokenizer, query, answer, demon_list[i:], system, proj_tokens, eos, query_format, demon_proj)
            if input_ids.shape[-1] > max_len:
                if i == len(demon_list):
                    logging.info(f"[WARNING] zero-shot overflow!")
            else:
                if i != 0:
                    logging.info(f"[WARNING] {len(demon_list) - i}-shot overflow!")
                break
    else:
        for i in range(len(demon_list),-1,-1):
            input_ids, input_mask = sv_format(tokenizer, query, answer, demon_list[:i], system, proj_tokens, eos, query_format, demon_proj)
            if input_ids.shape[-1] > max_len:
                if i == 0:
                    logging.info(f"[WARNING] zero-shot overflow!")
            else:
                if i != len(demon_list):
                    logging.info(f"[WARNING] {i+1}-shot overflow!")
                break
    return input_ids, input_mask


class ModelBase:
    def __init__(self, config):
        self.model, self.tokenizer, self.device = self.load(config['model_path'], config)

    def load(self, model_path, config):
        tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side='left', padding_side='left', use_fast=False)
        if config.get("model_max_length") is not None:
            tokenizer.model_max_length = config.get("model_max_length")

        logging.info(f'loading model from: {model_path}')

        if config.get("device") is None:
            device = 0
        else:
            device = eval(config.get("device"))

        torch_dtype = torch.float32
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype).to(device)
        model.to(device)
        model = model.eval()

        logging.info(f'loading {type(model)} model done')

        return model, tokenizer, device


def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values:
        past_key_values_length = past_key_values[0][0].shape[2]
        input_ids = input_ids[:, past_key_values_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, past_key_values_length: ]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs

LlamaForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation
MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation