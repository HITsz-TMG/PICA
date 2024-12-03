import copy
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from baukit import TraceDict
from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from transformers import LlamaForCausalLM
from utils import ModelBase, intervention_mode2list, sv_format_length
from typing import List, Iterable
from state_vector_extract import extract_icl_sv, multi_extract_icl_sv

GLOBAL_USE_CACHE=True

class StoppingCriteriaSub(StoppingCriteria):
    # only for beam 1 search
    def __init__(self, stop_seqs:List[torch.Tensor]=[], input_len=0):
        super().__init__()
        self.stop_seqs = stop_seqs # stop ids
        self.input_len = input_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop_ids in self.stop_seqs:
            if input_ids.shape[1] - self.input_len < len(stop_ids): continue
            stop_count = (stop_ids != input_ids[0, -len(stop_ids):]).sum()
            if stop_count == 0:
                return True
        return False

# class NoRepeatNGramLogitsProcessor(StoppingCriteria):
#
#     def __init__(self, ngram_size: int):
#         if not isinstance(ngram_size, int) or ngram_size <= 0:
#             raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
#         self.ngram_size = ngram_size
#
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
#         num_batch_hypotheses = scores.shape[0]
#         cur_len = input_ids.shape[-1]
#         banned_batch_tokens = self._calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)
#
#         if len(banned_batch_tokens) > 0:
#             return True
#         return False
#
#     def _get_ngrams(self, ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
#         generated_ngrams = [{} for _ in range(num_hypos)]
#         for idx in range(num_hypos):
#             gen_tokens = prev_input_ids[idx].tolist()
#             generated_ngram = generated_ngrams[idx]
#             for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
#                 prev_ngram_tuple = tuple(ngram[:-1])
#                 generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
#         return generated_ngrams
#
#     def _get_generated_ngrams(self, banned_ngrams, prev_input_ids, ngram_size, cur_len):
#         # Before decoding the next token, prevent decoding of ngrams that have already appeared
#         start_idx = cur_len + 1 - ngram_size
#         ngram_idx = tuple(prev_input_ids[start_idx:cur_len].tolist())
#         return banned_ngrams.get(ngram_idx, [])
#
#     def _calc_banned_ngram_tokens(
#             self, ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
#     ) -> List[Iterable[int]]:
#         """Copied from fairseq for no_repeat_ngram in beam_search"""
#         if cur_len + 1 < ngram_size:
#             # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
#             return [[] for _ in range(num_hypos)]
#
#         generated_ngrams = self._get_ngrams(ngram_size, prev_input_ids, num_hypos)
#
#         banned_tokens = [
#             self._get_generated_ngrams(generated_ngrams[hypo_idx], prev_input_ids[hypo_idx], ngram_size, cur_len)
#             for hypo_idx in range(num_hypos)
#         ]
#         return banned_tokens

class ModelFramework(ModelBase):
    def __init__(self, config):
        super().__init__(config)

    def generate_chche(self, demon:List=None, format_dict={'eos': '\n\n', 'proj_tokens': '→'}):
        input_ids, input_mask = sv_format_length(self.tokenizer, None, None, demon, max_len=self.tokenizer.model_max_length, **format_dict)
        input_ids = input_ids.to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, return_dict=True)
        return output['past_key_values']

    def prior_ICL_generation(self, question, demon: List, prior_token_num, interventation_layer ,state_vector=None, past_key_values=None, gen_kwargs: dict = None, format_dict={'eos': '\n\n', 'proj_tokens': '→'}, return_generate_time=False):
        if gen_kwargs is None: gen_kwargs = {}
        else: gen_kwargs = copy.deepcopy(gen_kwargs)

        # ------------ first generation ------------
        first_stage_gen_kwargs = copy.deepcopy(gen_kwargs)
        if past_key_values is not None:
            first_stage_gen_kwargs["past_key_values"] = past_key_values

        input_ids, input_mask = sv_format_length(self.tokenizer, question, None, demon, max_len=self.tokenizer.model_max_length, **format_dict)
        input_ids = input_ids.to(self.device)

        if prior_token_num != 0:
            first_stage_gen_kwargs['max_new_tokens'] = prior_token_num
            if 'max_new_tokens' in gen_kwargs: gen_kwargs['max_new_tokens'] = gen_kwargs['max_new_tokens'] - prior_token_num
            if 'max_length' not in first_stage_gen_kwargs: first_stage_gen_kwargs['max_length'] = self.tokenizer.model_max_length
            first_stage_gen_kwargs['max_length'] = min(first_stage_gen_kwargs['max_length'],input_ids.shape[-1] + first_stage_gen_kwargs.pop("max_new_tokens"))
            if 'eos_token_id' not in first_stage_gen_kwargs: first_stage_gen_kwargs['eos_token_id'] = [self.tokenizer.eos_token_id]
            eos_token_id = first_stage_gen_kwargs.pop('eos_token_id')
            if len(eos_token_id):
                first_stage_gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StoppingCriteriaSub(
                    stop_seqs=[torch.tensor(eos_token_id, dtype=torch.long, device=input_ids.device)],
                    input_len=input_ids.shape[1])]
                )

            stime1 = time.time()
            with torch.no_grad():
                outputs = self.model.generate(input_ids=input_ids,
                                              use_cache=GLOBAL_USE_CACHE,
                                              return_dict_in_generate=True,
                                              **first_stage_gen_kwargs)
            etime1 = time.time()


            sequences = outputs.sequences
            is_first_stage_end = sequences[0, -len(eos_token_id)].tolist() == eos_token_id
            prior_ids = sequences[:, input_ids.shape[1]:]

            inp_sequences = self.tokenizer.decode(sequences[0, :input_ids.shape[1]].tolist(), skip_special_tokens=False)
            gen_sequences = sequences[0, input_ids.shape[1]:].tolist()
            while gen_sequences[-len(eos_token_id):] == eos_token_id: gen_sequences = gen_sequences[:-len(eos_token_id)]
            gen_sequences = self.tokenizer.decode(gen_sequences, skip_special_tokens=False)

            # print('[输入1]')
            # print(inp_sequences)
            print('[输出1]')
            print(gen_sequences)

            if is_first_stage_end:
                print('[WARNING] First stage end')
                if return_generate_time:
                    return inp_sequences, gen_sequences, etime1-stime1
                return inp_sequences, gen_sequences
        else:
            prior_ids = torch.tensor([[]], dtype=input_ids.dtype, device=input_ids.device)

        # state vector
        if state_vector is None:
            if past_key_values is not None:
                input_ids = input_ids[:, past_key_values[0][0].shape[2]:]
                input_mask = input_mask[past_key_values[0][0].shape[2]:]

            layer_hook_names = intervention_mode2list("Wo", interventation_layer, prefix="")

            with torch.no_grad():
                with TraceDict(self.model, layers=layer_hook_names, clone=False, detach=False, retain_input=False, retain_output=True) as activations_td:
                    logits = self.model(input_ids, past_key_values=past_key_values).logits.cpu()
            hook_input = {l: activations_td[l].output[0].cpu() for l in layer_hook_names}

            proj_name = f"project"
            indices = torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == proj_name],dtype=torch.long)
            state_vector = {k: v[indices] for k, v in hook_input.items()}


        # second generation
        input_ids, input_mask = sv_format_length(self.tokenizer, question, None, None, max_len=self.tokenizer.model_max_length, **format_dict)
        input_ids = input_ids.to(self.device)
        input_ids = torch.cat((input_ids, prior_ids), dim=-1)

        if 'max_length' not in gen_kwargs: gen_kwargs['max_length'] = self.tokenizer.model_max_length
        if 'max_new_tokens' in gen_kwargs: gen_kwargs['max_length'] = min(gen_kwargs['max_length'], input_ids.shape[-1] + gen_kwargs.pop("max_new_tokens"))
        if 'eos_token_id' not in gen_kwargs: gen_kwargs['eos_token_id'] = [self.tokenizer.eos_token_id]
        eos_token_id = gen_kwargs.pop('eos_token_id')
        if len(eos_token_id):
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StoppingCriteriaSub(
            stop_seqs=[torch.tensor(eos_token_id, dtype=torch.long, device=input_ids.device)],
            input_len=input_ids.shape[1])]
        )

        layer_indices = torch.cat(
            [
                torch.tensor([i for i in range(len(input_mask)) if input_mask[i] == wp], dtype=torch.long)
                for wp in ["project"]
            ], dim=0
        )
        layer_hook_names = list(state_vector.keys())
        intervention_fn = self.intervention_function(state_vector, layer_indices, layer_hook_names)

        stime2 = time.time()
        with torch.no_grad():
            with TraceDict(self.model, layers=layer_hook_names, clone=False, detach=False, retain_input=False,retain_output=False, edit_output=intervention_fn) as activations_td:
                outputs = self.model.generate(input_ids=input_ids,
                                              use_cache=GLOBAL_USE_CACHE,
                                              return_dict_in_generate=True,
                                              **gen_kwargs)
        etime2 = time.time()

        sequences = outputs.sequences
        inp_sequences = self.tokenizer.decode(sequences[0, :input_ids.shape[1] - prior_ids.shape[1]].tolist(), skip_special_tokens=False)
        gen_sequences = sequences[0, input_ids.shape[1] - prior_ids.shape[1]:].tolist()
        while gen_sequences[-len(eos_token_id):] == eos_token_id: gen_sequences = gen_sequences[:-len(eos_token_id)]
        gen_sequences = self.tokenizer.decode(gen_sequences, skip_special_tokens=False)

        # print('[输入2]')
        # print(inp_sequences)
        print('[输出2]')
        print(gen_sequences)

        if return_generate_time:
            return inp_sequences, gen_sequences, etime1 - stime1 + etime2 - stime2
        return inp_sequences, gen_sequences

    def pure_prior_ICL_generation(self, question, demon: List, prior_token_num, past_key_values=None, gen_kwargs: dict = None, format_dict={'eos': '\n\n', 'proj_tokens': '→'}, return_generate_time=False):
        if gen_kwargs is None: gen_kwargs = {}
        else: gen_kwargs = copy.deepcopy(gen_kwargs)

        # ------------ first generation ------------
        first_stage_gen_kwargs = copy.deepcopy(gen_kwargs)
        if past_key_values is not None:
            first_stage_gen_kwargs["past_key_values"] = past_key_values

        input_ids, input_mask = sv_format_length(self.tokenizer, question, None, demon, max_len=self.tokenizer.model_max_length, **format_dict)
        input_ids = input_ids.to(self.device)

        first_stage_gen_kwargs['max_new_tokens'] = prior_token_num
        if 'max_new_tokens' in gen_kwargs: gen_kwargs['max_new_tokens'] = gen_kwargs['max_new_tokens'] - prior_token_num
        if 'max_length' not in first_stage_gen_kwargs: first_stage_gen_kwargs['max_length'] = self.tokenizer.model_max_length
        first_stage_gen_kwargs['max_length'] = min(first_stage_gen_kwargs['max_length'],input_ids.shape[-1] + first_stage_gen_kwargs.pop("max_new_tokens"))
        if 'eos_token_id' not in first_stage_gen_kwargs: first_stage_gen_kwargs['eos_token_id'] = [self.tokenizer.eos_token_id]
        eos_token_id = first_stage_gen_kwargs.pop('eos_token_id')
        if len(eos_token_id):
            first_stage_gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StoppingCriteriaSub(
            stop_seqs=[torch.tensor(eos_token_id, dtype=torch.long, device=input_ids.device)],
            input_len=input_ids.shape[1])]
        )

        stime1 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids,
                                          use_cache=GLOBAL_USE_CACHE,
                                          return_dict_in_generate=True,
                                          **first_stage_gen_kwargs)
        etime1 = time.time()

        sequences = outputs.sequences
        is_first_stage_end = sequences[0, -len(eos_token_id)].tolist() == eos_token_id
        prior_ids = sequences[:, input_ids.shape[1]:]

        inp_sequences = self.tokenizer.decode(sequences[0, :input_ids.shape[1]].tolist(), skip_special_tokens=False)
        gen_sequences = sequences[0, input_ids.shape[1]:].tolist()
        while gen_sequences[-len(eos_token_id):] == eos_token_id: gen_sequences = gen_sequences[:-len(eos_token_id)]
        gen_sequences = self.tokenizer.decode(gen_sequences, skip_special_tokens=False)

        # print('[输入1]')
        # print(inp_sequences)
        print('[输出1]')
        print(gen_sequences)

        if is_first_stage_end:
            print('[WARNING] First stage end')
            if return_generate_time:
                return inp_sequences, gen_sequences, etime1-stime1
            return inp_sequences, gen_sequences

        # second generation
        input_ids, input_mask = sv_format_length(self.tokenizer, question, None, None, max_len=self.tokenizer.model_max_length, **format_dict)
        input_ids = input_ids.to(self.device)
        input_ids = torch.cat((input_ids, prior_ids), dim=-1)

        if 'max_length' not in gen_kwargs: gen_kwargs['max_length'] = self.tokenizer.model_max_length
        if 'max_new_tokens' in gen_kwargs: gen_kwargs['max_length'] = min(gen_kwargs['max_length'], input_ids.shape[-1] + gen_kwargs.pop("max_new_tokens"))
        if 'eos_token_id' not in gen_kwargs: gen_kwargs['eos_token_id'] = [self.tokenizer.eos_token_id]
        eos_token_id = gen_kwargs.pop('eos_token_id')
        if len(eos_token_id):
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StoppingCriteriaSub(
                stop_seqs=[torch.tensor(eos_token_id, dtype=torch.long, device=input_ids.device)],
                input_len=input_ids.shape[1])]
            )

        stime2 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids,
                                          use_cache=GLOBAL_USE_CACHE,
                                          return_dict_in_generate=True,
                                          **gen_kwargs)
        etime2 = time.time()

        sequences = outputs.sequences
        inp_sequences = self.tokenizer.decode(sequences[0, :input_ids.shape[1] - prior_ids.shape[1]].tolist(), skip_special_tokens=False)
        gen_sequences = sequences[0, input_ids.shape[1] - prior_ids.shape[1]:].tolist()
        while gen_sequences[-len(eos_token_id):] == eos_token_id: gen_sequences = gen_sequences[:-len(eos_token_id)]
        gen_sequences = self.tokenizer.decode(gen_sequences, skip_special_tokens=False)

        print('[输出2]')
        print(gen_sequences)
        if return_generate_time:
            return inp_sequences, gen_sequences, etime1 - stime1 + etime2 - stime2
        return inp_sequences, gen_sequences

    def ICL_generation(self, question, demon: List, past_key_values=None, gen_kwargs: dict = None, format_dict={'eos': '\n\n', 'proj_tokens': '→'}, return_generate_time=False):
        if gen_kwargs is None: gen_kwargs = {}
        else: gen_kwargs = copy.deepcopy(gen_kwargs)

        if past_key_values is not None:
            gen_kwargs["past_key_values"] = past_key_values

        input_ids, input_mask = sv_format_length(self.tokenizer, question, None, demon, max_len=self.tokenizer.model_max_length, **format_dict)
        input_ids = input_ids.to(self.device)

        if 'max_length' not in gen_kwargs: gen_kwargs['max_length'] = self.tokenizer.model_max_length
        if 'max_new_tokens' in gen_kwargs: gen_kwargs['max_length'] = min(gen_kwargs['max_length'],input_ids.shape[-1] + gen_kwargs.pop("max_new_tokens"))
        if 'eos_token_id' not in gen_kwargs: gen_kwargs['eos_token_id'] = [self.tokenizer.eos_token_id]
        eos_token_id = gen_kwargs.pop('eos_token_id')
        if len(eos_token_id):
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StoppingCriteriaSub(
            stop_seqs=[torch.tensor(eos_token_id, dtype=torch.long, device=input_ids.device)],
            input_len=input_ids.shape[1])]
        )

        stime = time.time()
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids,
                                          use_cache=GLOBAL_USE_CACHE,
                                          return_dict_in_generate=True,
                                          **gen_kwargs)
        etime = time.time()

        sequences = outputs.sequences

        inp_sequences = self.tokenizer.decode(sequences[0, :input_ids.shape[1]].tolist(), skip_special_tokens=False)
        gen_sequences = sequences[0, input_ids.shape[1]:].tolist()
        while gen_sequences[-len(eos_token_id):] == eos_token_id: gen_sequences = gen_sequences[:-len(eos_token_id)]
        gen_sequences = self.tokenizer.decode(gen_sequences, skip_special_tokens=False)

        if return_generate_time:
            return inp_sequences, gen_sequences, etime-stime
        return inp_sequences, gen_sequences

    def intervention_function(self, state_vector, layer_indices, layer_hook_names, lam_old=0, lam_new=1):
        def merge(output, layer_name):
            nonlocal layer_hook_names
            if layer_name in layer_hook_names:
                activation = state_vector[layer_name].to(output.device) # bsz, head_num, vitural_token_num, head_dim
                activation = activation.unsqueeze(0).expand(output.shape[0],-1,-1)
                output[:, layer_indices] = lam_old * output[:, layer_indices] + lam_new * activation
                layer_hook_names.remove(layer_name)
            return output
        layer_hook_names = copy.deepcopy(layer_hook_names)
        return merge

