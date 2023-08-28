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

    def _generate_one_token(self, tokens, input_text_mask,
                            cur_pos, prev_pos,
                            temperature,
                            top_p):
        logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        return tokens

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
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
                print(f"{i}-th decoding phase")
            tokens = self._generate_one_token_fn(tokens, input_text_mask, cur_pos, prev_pos, temperature, top_p)
            prev_pos = cur_pos

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
