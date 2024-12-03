import argparse
import copy
import random
import sys
import json
import os
from typing import List
import peft
from peft import PeftModel, TaskType
from tqdm import tqdm
import torch
import transformers
from baukit import TraceDict
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM
import logging
from utils import set_seed, sv_format_length, ModelBase, intervention_mode2list

set_seed(233)


def extract_icl_sv(
        tokenizer,
        model,
        dummy:str,
        dev:List,
        interventation_layer: int,
        example_num=-1,
        save_path=None,
        format_dict={'eos': '\n\n', 'proj_tokens': '→'},
        extract_mode='KV'
):
    model.eval()
    demon = dev[: example_num] if example_num != -1 else dev
    input_ids, input_mask = sv_format_length(tokenizer, dummy, None, demon, max_len=tokenizer.model_max_length, **format_dict)

    layer_hook_names = intervention_mode2list(extract_mode, interventation_layer, prefix="")

    with torch.no_grad():
        with TraceDict(model, layers=layer_hook_names, clone=True, detach=True, retain_input=False, retain_output=True) as activations_td:
            logits = model(input_ids.to(model.device)).logits.cpu()
    hook_input = {l: activations_td[l].output[0].cpu() for l in layer_hook_names}

    proj_name = f"project"
    indices = torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == proj_name], dtype=torch.long)
    state_vector = {k: v[indices] for k, v in hook_input.items()}

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        torch.save(state_vector, os.path.join(save_path, f'state_vector.pt'))

    return state_vector



def batch_extract_icl_sv(
        tokenizer,
        model,
        dummies: List[str],
        devs: List[List],
        interventation_layer: int,
        example_num=-1,
        save_path=None,
        format_dict={'eos': '\n\n', 'proj_tokens': '→'},
        extract_mode='KV'
):
    model.eval()
    demons = [(dev[: example_num] if example_num != -1 else dev) for dev in devs]
    input_ids_list = []
    input_mask_list = []
    original_len = []
    max_len = 0
    batch_size = len(dummies)
    for dummy, demon in zip(dummies, demons):
        input_ids, input_mask = sv_format_length(tokenizer, dummy, None, demon, max_len=tokenizer.model_max_length, **format_dict)
        input_ids_list.append(input_ids[0])
        input_mask_list.append(input_mask)
        original_len.append(len(input_ids[0]))
        max_len = max(max_len, len(input_ids[0]))

    # pad
    input_ids_list = [torch.cat((input_ids_list[i], torch.zeros((max_len - original_len[i]), dtype=torch.long)), dim=0) for i in range(len(input_ids_list))]
    input_ids_list = torch.stack(input_ids_list, dim=0)

    layer_hook_names = intervention_mode2list(extract_mode, interventation_layer, prefix="")

    with torch.no_grad():
        with TraceDict(model, layers=layer_hook_names, clone=True, detach=True, retain_input=False,
                       retain_output=True) as activations_td:
            logits = model(input_ids_list.to(model.device)).logits.cpu()

    state_vectors = []
    for b in range(batch_size):
        hook_input = {l: activations_td[l].output[b].cpu() for l in layer_hook_names}
        input_mask = input_mask_list[b]
        proj_name = f"project"
        indices = torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == proj_name], dtype=torch.long)
        state_vector = {k: v[indices] for k, v in hook_input.items()}
        state_vectors.append(state_vector)

    return state_vectors

def multi_extract_icl_sv(tokenizer, model, dummys, devs, interventation_layer:int, example_num=None, save_path=None, format_dict={'eos': '\n\n', 'proj_tokens': '→'}, extract_mode='KV', batch_size=-1):
    state_vectors = []
    if not isinstance(dummys, List):
        dummys = [dummys]
    if not isinstance(devs, List):
        devs = [devs]

    dummy_batch = []
    dev_batch = []
    for dummy in dummys:
        for dev in devs:
            dummy_batch.append(dummy)
            dev_batch.append(dev)

    if batch_size == -1: batch_size = len(dummy_batch)

    for i in range(0, len(dummy_batch), batch_size):
        state_vectors += batch_extract_icl_sv(tokenizer=tokenizer, model=model, dummies=dummy_batch[i:i+batch_size],
                                              devs=dev_batch[i:i+batch_size], interventation_layer=interventation_layer,
                                              example_num=example_num, save_path=save_path, format_dict=format_dict,
                                              extract_mode=extract_mode)
    return avg_sv(state_vectors)

def avg_sv(raw_task_vector_list):
    task_vector_list = {}
    for tv_name in raw_task_vector_list[0]:
        tv = [raw_task_vector_list[i][tv_name] for i in range(len(raw_task_vector_list))]
        tv = torch.stack(tv, dim=0).mean(dim=0)
        task_vector_list[tv_name] = tv
    return task_vector_list


