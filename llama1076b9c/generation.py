# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch
import os

from llama1076b9c.tokenizer import Tokenizer
from llama1076b9c.model import Transformer

USE_TORCH_DYNAMO = bool(int(os.environ.get('USE_TORCH_DYNAMO', 1)))


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._generate_one_token_fn = self._generate_one_token
        if USE_TORCH_DYNAMO:
            from typing import List
            def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
                print("custom backend called with FX graph:")
                gm.graph.print_tabular()
                return gm.forward

            # Reset since we are using a different backend.
            torch._dynamo.reset()
            self._generate_one_token_fn = torch.compile(self._generate_one_token_fn, backend=custom_backend)

    def _generate_one_token(self, tokens, input_tokens, prev_pos, input_text_mask,
                            cur_pos_tensor, input_pos_tensor,
                            output_pos_tensor, cache_kvs, temperature_tensor,
                            top_p_tensor, with_temp):
        logits, cache_kvs = self.model(input_tokens, prev_pos, input_pos_tensor, output_pos_tensor, cache_kvs)
        if with_temp:
            probs = torch.softmax(logits / temperature_tensor, dim=-1)
            next_token = sample_top_p(probs, top_p_tensor)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        input_text_mask_tmp = input_text_mask.index_select(1, cur_pos_tensor).squeeze(dim=1)
        tokens_tmp = tokens.index_select(1, cur_pos_tensor).squeeze(dim=1)
        next_token = torch.where(
            input_text_mask_tmp, tokens_tmp, next_token
        )
        # tokens[:, cur_pos] = next_token
        next_token = next_token.unsqueeze(1)
        tokens.index_copy_(1, cur_pos_tensor, next_token)
        # prepare for the next iteration
        input_pos_tensor = cur_pos_tensor.unsqueeze(0)
        cur_pos_tensor = cur_pos_tensor + 1
        output_pos_tensor = cur_pos_tensor - 1
        input_tokens = tokens.index_select(1, input_pos_tensor)

        return tokens, input_tokens, cur_pos_tensor, input_pos_tensor, output_pos_tensor, cache_kvs


    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        device = 'cuda'
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((params.max_batch_size, params.max_seq_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        tokens = tokens.to(device)

        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size

        # Passing tensors instead of floats into self._generate_one_token_fn,
        # so that different values would not trigger compilations of new graphs
        temperature_tensor = torch.tensor(float(temperature)).to(device)
        top_p_tensor = torch.tensor(float(top_p)).to(device)
        with_temp = temperature > 0

        cache_kvs = self.model.cache_kvs

        prev_pos = 0
        i = 0
        print(f"total_len: {total_len}")
        for cur_pos in range(start_pos, total_len):
            print("\n")
            print(f"cur_pos: {cur_pos}")
            i += 1
            if i == 1:
                print(f"{i}-th pre-fill-phase")
            else:
                print(f"{i-1}-th decoding phase")
            prev_pos_tensor = torch.tensor(prev_pos).to(device)
            cur_pos_tensor = torch.tensor(cur_pos).to(device)
            output_pos_tensor = torch.tensor(cur_pos - 1).to(device)
            input_pos_tensor = torch.arange(prev_pos, cur_pos).to(device)
            input_tokens = tokens.index_select(1, input_pos_tensor)
            tokens, input_tokens, cur_pos_tensor, input_pos_tensor, output_pos_tensor, cache_kvs \
                = self._generate_one_token_fn(
                    tokens, input_tokens, prev_pos_tensor, input_text_mask, cur_pos_tensor,
                    input_pos_tensor, output_pos_tensor, cache_kvs,
                    temperature_tensor, top_p_tensor, with_temp
                )
            prev_pos = cur_pos
        self.model.cache_kvs = cache_kvs

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
