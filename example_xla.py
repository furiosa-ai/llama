# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
from pathlib import Path

from llama import ModelArgs, Transformer, Tokenizer, LLaMA
from llama.xla_model_parallel import (
    get_model_parallel_rank,
    get_model_parallel_world_size,
)

import os

USE_CUDA = bool(int(os.environ.get("USE_CUDA", 0)))
USE_XLA = bool(int(os.environ.get("USE_XLA", 0)))
USE_TORCH_DYNAMO = bool(int(os.environ.get("USE_TORCH_DYNAMO", 1)))
GPU_NUM_DEVICES = int(os.environ.get("GPU_NUM_DEVICES", 0))


# Some how xla init will slow down the CUDA speed.
if USE_XLA:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp


def setup_model_parallel() -> Tuple[int, int]:
    # assuming model parallelism over the whole world size
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()

    # seed must be the same in all processes
    torch.manual_seed(1)
    device = xm.xla_device()
    xm.set_rng_state(1, device=device)
    return rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    device: torch.device,
    dim: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
    quant: bool = False,
) -> LLaMA:
    start_time = time.time()
    print("Loading")
    if ckpt_dir:
        # load checkpoint if ckpt_dir is provided
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert world_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
        ckpt_path = checkpoints[rank]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
    else:
        params = {
            "dim": dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "quant": quant,
        }

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    if USE_CUDA:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.BFloat16Tensor)
    model = Transformer(model_args)
    if ckpt_dir:
        model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    for i in range(len(model.cache_kvs)):
        model.cache_kvs[i] = tuple(t.to(device) for t in model.cache_kvs[i])
    torch.set_default_tensor_type(torch.FloatTensor)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    ckpt_dir: str = "",
    dim: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
    quant: bool = False,
    n_times: int = 3,
):
    rank, world_size = setup_model_parallel()
    if rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir,
        tokenizer_path,
        rank,
        world_size,
        max_seq_len,
        max_batch_size,
        xm.xla_device(),
        dim,
        n_layers,
        n_heads,
        quant,
    )

    def random_str_gen(N):
        import random
        import string

        return " ".join(
            random.choice(string.ascii_uppercase + string.digits) for _ in range(N)
        )

    # warmup stage
    # print("pre-compile graphs with buckets = [128, 256, 384, 512]")
    # dummy_prompts = [
    #     [random_str_gen(64)], # for 128 bucket
    #     [random_str_gen(128)],# for 256 bucket
    #     [random_str_gen(256)],# for 384 bucket
    #     [random_str_gen(384)],# for 512 bucket
    # ]
    # for dummy in dummy_prompts:
    #     with torch.no_grad():
    #         generator.generate(
    #             dummy, 4, xm.xla_device(), temperature=temperature, top_p=top_p
    #         )
    # print("warmup done.\n")

    prompts = [
        # # 1519 tokens ==> 512 bucket
        # [
        #     """
        # Mathematics is an area of knowledge that includes the topics of numbers, formulas and related structures, shapes and the spaces in which they are contained, and quantities and their changes. These topics are represented in modern mathematics with the major subdisciplines of number theory,[1] algebra,[2] geometry,[1] and analysis,[3][4] respectively. There is no general consensus among mathematicians about a common definition for their academic discipline.
        # Most mathematical activity involves the discovery of properties of abstract objects and the use of pure reason to prove them. These objects consist of either abstractions from nature or—in modern mathematics—entities that are stipulated to have certain properties, called axioms. A proof consists of a succession of applications of deductive rules to already established results. These results include previously proved theorems, axioms, and—in case of abstraction from nature—some basic properties that are considered true starting points of the theory under consideration.[5]
        # Mathematics is essential in the natural sciences, engineering, medicine, finance, computer science and the social sciences. Although mathematics is extensively used for modeling phenomena, the fundamental truths of mathematics are independent from any scientific experimentation. Some areas of mathematics, such as statistics and game theory, are developed in close correlation with their applications and are often grouped under applied mathematics. Other areas are developed independently from any application (and are therefore called pure mathematics), but often later find practical applications.[6][7] The problem of integer factorization, for example, which goes back to Euclid in 300 BC, had no practical application before its use in the RSA cryptosystem, now widely used for the security of computer networks.
        # Historically, the concept of a proof and its associated mathematical rigour first appeared in Greek mathematics, most notably in Euclid's Elements.[8] Since its beginning, mathematics was essentially divided into geometry and arithmetic (the manipulation of natural numbers and fractions), until the 16th and 17th centuries, when algebra[a] and infinitesimal calculus were introduced as new areas. Since then, the interaction between mathematical innovations and scientific discoveries has led to a rapid lockstep increase in the development of both.[9] At the end of the 19th century, the foundational crisis of mathematics led to the systematization of the axiomatic method,[10] which heralded a dramatic increase in the number of mathematical areas and their fields of application. The contemporary Mathematics Subject Classification lists more than 60 first-level areas of mathematics.
        # The word mathematics comes from Ancient Greek máthēma (μάθημα), meaning "that which is learnt",[11] "what one gets to know", hence also "study" and "science". The word came to have the narrower and more technical meaning of "mathematical study" even in Classical times.[12] Its adjective is mathēmatikós (μαθηματικός), meaning "related to learning" or "studious", which likewise further came to mean "mathematical".[13] In particular, mathēmatikḗ tékhnē (μαθηματικὴ τέχνη; Latin: ars mathematica) meant "the mathematical art".[11]
        # Similarly, one of the two main schools of thought in Pythagoreanism was known as the mathēmatikoi (μαθηματικοί)—which at the time meant "learners" rather than "mathematicians" in the modern sense. The Pythagoreans were likely the first to constrain the use of the word to just the study of arithmetic and geometry. By the time of Aristotle (384–322 BC) this meaning was fully established.[14]
        # In Latin, and in English until around 1700, the term mathematics more commonly meant "astrology" (or sometimes "astronomy") rather than "mathematics"; the meaning gradually changed to its present one from about 1500 to 1800. This change has resulted in several mistranslations: For example, Saint Augustine's warning that Christians should beware of mathematici, meaning "astrologers", is sometimes mistranslated as a condemnation of mathematicians.[15]
        # The apparent plural form in English goes back to the Latin neuter plural mathematica (Cicero), based on the Greek plural ta mathēmatiká (τὰ μαθηματικά) and means roughly "all things mathematical", although it is plausible that English borrowed only the adjective mathematic(al) and formed the noun mathematics anew, after the pattern of physics and metaphysics, inherited from Greek.[16] In English, the noun mathematics takes a singular verb. It is often shortened to maths or, in North America, math.[17]
        # Before the Renaissance, mathematics was divided into two main areas: arithmetic, regarding the manipulation of numbers, and geometry, regarding the study of shapes.[18] Some types of pseudoscience, such as numerology and astrology, were not then clearly distinguished from mathematics.[19]
        # During the Renaissance, two more areas appeared. Mathematical notation led to algebra which, roughly speaking, consists of the study and the manipulation of formulas. Calculus, consisting of the two subfields differential calculus and integral calculus, is the study of continuous functions, which model the typically nonlinear relationships between varying quantities, as represented by variables. This division into four main areas–arithmetic, geometry, algebra, calculus[20]–endured until the end of the 19th century. Areas such as celestial mechanics and solid mechanics were then studied by mathematicians, but now are considered as belonging to physics.[21] The subject of combinatorics has been studied for much of recorded history, yet did not become a separate branch of mathematics until the seventeenth century.[22]
        # At the end of the 19th century, the foundational crisis in mathematics and the resulting systematization of the axiomatic method led to an explosion of new areas of mathematics.[23][10] The 2020 Mathematics Subject Classification contains no less than sixty-three first-level areas.[24] Some of these areas correspond to the older division, as is true regarding number theory (the modern name for higher arithmetic) and geometry. Several other first-level areas have "geometry" in their names or are otherwise commonly considered part of geometry. Algebra and calculus do not appear as first-level areas but are respectively split into several first-level areas. Other first-level areas emerged during the 20th century or had not previously been considered as mathematics, such as mathematical logic and foundations.[25]
        # """,
        # ],
        # 347 tokens ==> 384 bucket
        [
            "Tesla, Inc. (/ˈtɛslə/ TESS-lə or /ˈtɛzlə/ TEZ-lə[a]) is an American multinational automotive and clean energy company headquartered in Austin, Texas. Tesla designs and manufactures electric vehicles (cars and trucks), stationary battery energy storage devices from home to grid-scale, solar panels and solar shingles, and related products and services.\
        Tesla is one of the world's most valuable companies and, as of 2023, is the world's most valuable automaker. In 2022, the company led the battery electric vehicle market, with 18% share.\
        Its subsidiary Tesla Energy develops and is a major installer of photovoltaic systems in the United States and is one of the largest global suppliers of battery energy storage systems with 6.5 gigawatt-hours (GWh) installed in 2022.\
        Tesla was incorporated in July 2003 by Martin Eberhard and Marc Tarpenning as Tesla Motors. The company's name is a tribute to inventor and electrical engineer Nikola Tesla. In February 2004, via a $6.5 million investment, Elon Musk became the company's largest shareholder. He became CEO in 2008. Tesla's announced mission is to create products which help “accelerate the world’s transition to sustainable energy.”",
        ],
        # # 149 tokens => 256 bucket
        # [
        #     "Formula One (more commonly known as Formula 1 or F1) is the highest class of international racing for open-wheel single-seater formula racing cars sanctioned by the Fédération Internationale de l'Automobile (FIA). The FIA Formula One World Championship has been one of the premier forms of racing around the world since its inaugural season in 1950. The word formula in the name refers to the set of rules to which all participants' cars must conform.[1] A Formula One season consists of a series of races, known as Grands Prix. Grands Prix take place in multiple countries and continents around the world on either purpose-built circuits or closed public roads.",
        # ],
        # # 82 tokens ==> 128 bucket
        # [
        #     "Generative Pre-trained Transformer 3 (GPT-3) is a large language model released by OpenAI in 2020. Like its predecessor GPT-2, it is a decoder-only transformer model of deep neural network, which uses attention in place of previous recurrence- and convolution-based architectures.[2] Attention mechanisms allow",
        # ],
        # # 8 tokens ==> 128 bucket
        # ["I believe the meaning of life is",],
    ]

    for _ in range(n_times):
        with torch.no_grad():
            for prompt in prompts:
                results = generator.generate(
                    prompt, 256, xm.xla_device(), temperature=temperature, top_p=top_p
                )

        for result in results:
            print(result)
            print("\n==================================\n")

    print("XLA:GPU ENV INFO:")
    print(
        f"USE_XLA: {bool(USE_XLA)}, USE_CUDA: {bool(USE_CUDA)}, USE_TORCH_DYNAMO: {bool(USE_TORCH_DYNAMO)}, NUM_GPU(S): {GPU_NUM_DEVICES}"
    )
    print(
        f"Run LLaMA model in {ckpt_dir} for {n_times} prompts with {max_batch_size} max batch size(s)"
    )
    print(f"\t- mean latency: {sum(generator.latency_list[1:]) / n_times}(s)")
    print(
        f"\t- mean per-token latency: {sum(generator.per_token_latency_list[1:]) / n_times * 1000}(ms/token)"
    )


def _fn(
    idx,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    ckpt_dir: str = "",
    dim: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
    quant: bool = False,
    n_times: int = 3,
):
    main(
        tokenizer_path,
        temperature,
        top_p,
        max_seq_len,
        max_batch_size,
        ckpt_dir,
        dim,
        n_layers,
        n_heads,
        quant,
        n_times,
    )


def mp_main(
    mp: bool,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    ckpt_dir: str = "",
    dim: int = 4096,
    n_layers: int = 32,
    n_heads: int = 32,
    quant: bool = False,
    n_times: int = 3,
):
    if mp and USE_XLA:
        xmp.spawn(
            _fn,
            args=(
                tokenizer_path,
                temperature,
                top_p,
                max_seq_len,
                max_batch_size,
                ckpt_dir,
                dim,
                n_layers,
                n_heads,
                quant,
                n_times,
            ),
        )
    else:
        main(
            tokenizer_path,
            temperature,
            top_p,
            max_seq_len,
            max_batch_size,
            ckpt_dir,
            dim,
            n_layers,
            n_heads,
            quant,
            n_times,
        )


if __name__ == "__main__":
    fire.Fire(mp_main)
