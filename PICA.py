import copy
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os

import sys

import re
import string
import yaml
from tqdm import tqdm
import json
import requests
import argparse
import pandas as pd
import time
import transformers
from typing import List
from state_vector_extract import multi_extract_icl_sv
from Gen_Framework import ModelFramework

start_time = time.strftime("%Y_%m-%d-%H-%M-%S", time.localtime()).split('_')[-1]
print(f"start time: {start_time}, PID: {os.getpid()}")

class Evaluator:
    def __init__(self, config, evaluator: ModelFramework):
        self.config = config
        self.evaluator = evaluator

    def generate(self, data_path=None, default_dummy=None, default_demon=None, log=False, output_dir=None, resume_from_ckpt=False, use_cache=False):
        gen_kwargs = self.config['gen_kwargs']
        format_dict = self.config['format_dict']
        prior_token_num = self.config['prior_token_num']
        interventation_layer = self.config['interventation_layer']
        icl_mode = self.config['ICL_MODE']

        if data_path is None:
            data_path = self.config['data_path']
        data = json.load(open(data_path))

        if resume_from_ckpt:
            assert output_dir is not None
            results = json.load(open(os.path.join(output_dir, "model_outputs.json")))
            full_results = json.load(open(os.path.join(output_dir, "full_outputs.json")))
        else:
            results = []
            full_results = []

        if use_cache:
            past_key_values = self.evaluator.generate_chche(default_demon, format_dict)

        state_vector = None
        if default_dummy is not None and default_demon is not None:
            state_vector = multi_extract_icl_sv(tokenizer=self.evaluator.tokenizer,
                                                model=self.evaluator.model,
                                                dummys=default_dummy,
                                                devs=[default_demon],
                                                interventation_layer=interventation_layer,
                                                example_num=-1,
                                                save_path=None,
                                                format_dict=format_dict,
                                                extract_mode="Wo",
                                                batch_size=2)

        for x,d in enumerate(data):
            if x < len(results): continue
            print(f"******* {x} ********")
            instruction = d['instruction']
            result = {"instruction": instruction, "dataset": d['dataset'], "generator": "examples"}

            cur_past_key_values = None
            if 'demon' in d:
                if use_cache: logging.warning("!!! got custom demonstration, ignoring cache !!!")
                demon = d.get('demon')

            else:
                if use_cache: cur_past_key_values = past_key_values
                demon = default_demon

            if icl_mode == 'ICL':
                inp_sequences, gen_sequences = self.evaluator.ICL_generation(
                    question=instruction,
                    demon=demon,
                    past_key_values=cur_past_key_values,
                    gen_kwargs=gen_kwargs,
                    format_dict=format_dict
                )
            elif icl_mode == 'PICA':
                inp_sequences, gen_sequences = self.evaluator.prior_ICL_generation(
                    question=instruction,
                    demon=demon,
                    prior_token_num=prior_token_num,
                    interventation_layer=interventation_layer,
                    state_vector=state_vector,
                    past_key_values=cur_past_key_values,
                    gen_kwargs=gen_kwargs,
                    format_dict=format_dict
                )
            elif icl_mode == 'Prog': # progressive generation without ICL vector
                inp_sequences, gen_sequences = self.evaluator.pure_prior_ICL_generation(
                    question=instruction,
                    demon=demon,
                    prior_token_num=prior_token_num,
                    past_key_values=cur_past_key_values,
                    gen_kwargs=gen_kwargs,
                    format_dict=format_dict
                )
            else:
                raise KeyError

            gen_sequences = gen_sequences.strip()
            result["output"] = gen_sequences

            if log:
                print('-----------------------------')
                print(f'[输入]:\n{inp_sequences}')
                print(f'[输出]:\n{gen_sequences}')
                print('-----------------------------')

            results.append(result)
            full = copy.deepcopy(result)
            full['demon'] = demon
            full_results.append(full)

            if output_dir is not None:
                json.dump(results, open(os.path.join(output_dir, "model_outputs.json"), 'w'), ensure_ascii=False, indent=2)
                json.dump(full_results, open(os.path.join(output_dir, "full_outputs.json"), 'w'), ensure_ascii=False, indent=2)

        return results




def generate_eval(
        config,
        data_path=None,
        output_dir=None,
        log=False,
        default_dummy=None,
        default_demon=None,
        resume_from_ckpt=False,
        use_cache=False,
    ):
    if resume_from_ckpt:
        assert output_dir is not None
        config = json.load(open(os.path.join(output_dir, 'config.json')))

    if output_dir is not None and not resume_from_ckpt:
        os.makedirs(output_dir, exist_ok=True)
        json.dump(config, open(os.path.join(output_dir, 'config.json'), 'w'), ensure_ascii=False, indent=2)
        os.system(f'cp /GLOBALFS/hitsz_bthu_1/lzy/brainstrom/prior_token_generation/code/EVAL_alpaca_eval_ICL.py {output_dir}/EVAL_alpaca_eval_ICL.py')
        os.system(f'cp /GLOBALFS/hitsz_bthu_1/lzy/brainstrom/prior_token_generation/code/Gen_Framework.py {output_dir}/Gen_Framework.py')

    logging.info(config)

    evaluator = Evaluator(config["metric"], ModelFramework(config))
    result = evaluator.generate(data_path=data_path,
                                default_dummy=default_dummy,
                                default_demon=default_demon,
                                log=log,
                                output_dir=output_dir,
                                resume_from_ckpt=resume_from_ckpt,
                                use_cache=use_cache
                                )

    return result

def str2listint(s:str):
    if s is None: return []
    return [eval(i) for i in s.split(',')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # whether to print output string
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--resume_from_ckpt', action='store_true')
    # whether to use KV cache for faster generation (only false for speed testing)
    parser.add_argument('--use_cache', action='store_true')

    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--model_max_length', type=int, default=4096)
    parser.add_argument('--eos_token_id', type=str2listint, default=None)
    parser.add_argument('--num_beams', type=int, default=1)

    parser.add_argument('--prior_token_num', type=int, default=10)
    parser.add_argument('--interventation_layer', type=int, default=9)
    parser.add_argument('--icl_mode', default='PICA')
    parser.add_argument('--config', default=None)
    parser.add_argument('--dummy', default=None)

    parser.add_argument('--data_path', default=None)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    # args.data_path="/GLOBALFS/hitsz_bthu_1/lzy/brainstrom/data/alpaca_eval.json"
    # args.model_path="/GLOBALFS/hitsz_bthu_1/lzy/Model/llama-2-7b"
    # args.device="0"
    # args.log=True
    # args.output_dir="/GLOBALFS/hitsz_bthu_1/lzy/brainstrom/prior_token_generation/runs/debug"
    # args.max_new_tokens=4096
    # args.model_max_length=4096
    # args.eos_token_id=[13,28956,13]
    # args.prior_token_num=10
    # args.interventation_layer=9
    # args.icl_mode="PICA"
    # args.config="CFG6"
    # args.dummy="alpaca"



    if args.config == 'CFG6':
        from training_prompt import ICL_CONFIG_6 as ICL_CONFIG
    elif args.config == 'CFG5':
        from training_prompt import ICL_CONFIG_5 as ICL_CONFIG
    elif args.config == 'CFG4':
        from training_prompt import ICL_CONFIG_4 as ICL_CONFIG
    elif args.config == 'CFG3':
        from training_prompt import ICL_CONFIG_3 as ICL_CONFIG
    elif args.config == 'CFG0':
        from training_prompt import ICL_CONFIG_0 as ICL_CONFIG
    elif args.config == 'CFG_LLAMA_SFT':
        from training_prompt import SFT_CONFIG_CHAT as ICL_CONFIG
    elif args.config == 'CFG_MIS_SFT':
        from training_prompt import SFT_CONFIG_MISTRAL as ICL_CONFIG
    else:
        raise KeyError

    from training_prompt import ALPACA_DUMMY, JUST_DUMMY

    if args.dummy == 'alpaca':
        default_dummy = ALPACA_DUMMY
    elif args.dummy == 'just':
        default_dummy = JUST_DUMMY
    else:
        default_dummy = None

    generate_eval(
        log=True,
        output_dir=args.output_dir,
        config={
            "model_path": args.model_path,
            "device": args.device,
            "model_max_length": args.model_max_length,
            "metric": {
                "data_path": args.data_path,
                "gen_kwargs": {
                    "top_p": args.top_p,
                    "temperature": args.temperature,
                    "max_length": args.model_max_length,
                    "max_new_tokens": args.max_new_tokens,
                    "eos_token_id": args.eos_token_id,
                    "num_beams": args.num_beams,
                },
                "format_dict": ICL_CONFIG["format_dict"],
                "prior_token_num": args.prior_token_num,
                "interventation_layer": args.interventation_layer,
                "ICL_MODE": args.icl_mode
            }
        },
        default_dummy=default_dummy,
        default_demon=ICL_CONFIG['demons'],
        resume_from_ckpt=args.resume_from_ckpt,
        use_cache=args.use_cache,
        )

