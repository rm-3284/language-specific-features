import argparse
import math
import os
from math import inf
from pathlib import Path
from typing import TypedDict

import torch
from bracex import expand
from const import layer_to_index, model_choices, prompt_templates
from loader import load_dataset_specific
from nnsight import LanguageModel
from tqdm.auto import tqdm
from utils import (
    TqdmLoggingHandler,
    get_nested_attr,
    get_project_dir,
    set_deterministic,
)


class Args(TypedDict):
    model: str
    dataset_configs: list[str]
    layers: list[str]
    out_dir: Path
    out_path: Path | None
    hidden_dim: int | None


def parse_args() -> Args:
    parser = argparse.ArgumentParser(
        description="Collect activations from a dataset for particular layers and languages"
    )

    parser.add_argument(
        "model",
        help="model name",
        type=str,
        choices=model_choices,
    )

    parser.add_argument(
        "--dataset-configs",
        help='dataset configurations "dataset_name:config_name:split_name:start:end". Support bracex expansion',
        type=str,
        default=[],
        nargs="+",
    )

    parser.add_argument(
        "--layer",
        help="layer(s) to be processed. The values should be the path to the layer in the model. Support bracex expansion",
        type=str,
        default=[],
        nargs="+",
    )

    parser.add_argument(
        "--start",
        help="sample start index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end",
        help="sample end index",
        type=int,
        default=inf,
    )

    parser.add_argument(
        "--out-dir",
        help="output directory",
        type=Path,
        default=get_project_dir(),
    )

    parser.add_argument(
        "--out-path",
        help="output path",
        type=Path,
    )

    parser.add_argument(
        "--hidden-dim",
        help="hidden dimension of the model",
        type=int,
    )

    args = parser.parse_args()

    processed_layers = []

    for layer in args.layer:
        processed_layers.extend(expand(layer))

    processed_dataset_configs = []

    for dataset_config in args.dataset_configs:
        processed_dataset_configs.extend(expand(dataset_config))

    return {
        "model": args.model,
        "dataset_configs": processed_dataset_configs,
        "layers": processed_layers,
        "out_dir": args.out_dir,
        "out_path": args.out_path,
        "hidden_dim": args.hidden_dim,
    }


def count_activations(args, llm, dataset_name, lang, split_name, start, end):
    num_layers = len(args["layers"])
    hidden_dim = args["hidden_dim"]

    over_zero_token = torch.zeros((num_layers, hidden_dim))
    over_zero_example = torch.zeros((num_layers, hidden_dim))
    over_zero_total = torch.zeros((num_layers, hidden_dim))
    max_active_over_zero = torch.zeros((num_layers, hidden_dim))
    min_active_over_zero = torch.full((num_layers, hidden_dim), math.inf)
    num_examples = 0
    num_tokens = 0

    layers_modules = {layer: get_nested_attr(llm, layer) for layer in args["layers"]}

    dataset = load_dataset_specific(
        dataset_name,
        lang,
        split_name,
        int(start),
        int(end),
    )

    prompt_template = prompt_templates[dataset_name][lang]

    for row in tqdm(dataset, desc="Processing Samples", leave=False):
        prompt = prompt_template.format_map(row)

        with llm.trace(prompt):
            for layer in args["layers"]:
                layer_module = layers_modules[layer]
                layer_index = layer_to_index[layer]

                activations = layer_module.output

                over_zero_token[layer_index] += (activations > 0).sum(dim=(0, 1))
                over_zero_example[layer_index] += (activations > 0).any(dim=(0, 1))
                over_zero_total[layer_index] += activations.sum(dim=(0, 1))

                max_active_over_zero[layer_index] = torch.max(
                    max_active_over_zero[layer_index],
                    activations.max(dim=1).values.squeeze(0).round(decimals=3),
                )
                min_active_over_zero[layer_index] = torch.min(
                    min_active_over_zero[layer_index],
                    activations.min(dim=1).values.squeeze(0).round(decimals=3),
                )

        num_examples += 1
        tokenized_input_ids = llm.tokenizer(prompt, return_tensors="pt")["input_ids"]
        num_tokens += tokenized_input_ids.shape[1]

    return (
        over_zero_token,
        over_zero_example,
        over_zero_total,
        max_active_over_zero,
        min_active_over_zero,
        num_examples,
        num_tokens,
    )


@torch.inference_mode()
def main(args: Args):
    logger.info(f'Loading Model: {args["model"]}')

    llm = LanguageModel(args["model"], device_map="auto", dispatch=True)

    # Collect activations for each language
    for dataset_config in tqdm(args["dataset_configs"], desc="Processing languages"):
        dataset_name, lang, split_name, start, end = dataset_config.split(":")

        (
            over_zero_token,
            over_zero_example,
            over_zero_total,
            max_active_over_zero,
            min_active_over_zero,
            num_examples,
            num_tokens,
        ) = count_activations(args, llm, dataset_name, lang, split_name, start, end)

        output_dir = args["out_dir"] / args["out_path"] / dataset_name
        file_path = output_dir / f"{lang}.pt"

        os.makedirs(output_dir, exist_ok=True)

        # Save as a sparse tensor
        min_active_over_zero[min_active_over_zero == math.inf] = 0

        output = {
            "num_examples": num_examples,
            "num_tokens": num_tokens,
            "over_zero_token": over_zero_token.to_sparse(),
            "over_zero_example": over_zero_example.to_sparse(),
            "over_zero_total": over_zero_total.to_sparse(),
            "max_active_over_zero": max_active_over_zero.to_sparse(),
            "min_active_over_zero": min_active_over_zero.to_sparse(),
        }

        torch.save(output, file_path)


if __name__ == "__main__":
    logger = TqdmLoggingHandler.get_logger("activations_count")

    set_deterministic()

    args = parse_args()
    main(args)
