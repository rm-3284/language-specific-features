from pathlib import Path
from typing import TypedDict

import torch
from datasets import Dataset
from loader import load_sae
from nnsight import LanguageModel
#from sparsify import Sae
#from sparsify.sparse_coder import EncoderOutput
from depricated_classes import EncoderOutput
from sae_lens import SAE
from tqdm.auto import tqdm
from utils import get_device, get_nested_attr


def collect_activations(
    llm: LanguageModel, layers: list[str], prompt: str
) -> dict[str, list[torch.Tensor]]:

    layers_modules = {layer: get_nested_attr(llm, layer) for layer in layers}

    layers_activations = {layer: None for layer in layers}

    with llm.trace(prompt):
        for layer in layers:
            layers_activations[layer] = layers_modules[layer].output.cpu().save()

    layers_activations_processed = {
        layer: (
            activations[0].value
            if isinstance(activations, tuple)
            else activations.value
        )
        for layer, activations in layers_activations.items()
    }

    return layers_activations_processed


def sae_features_from_activations(
    activations_list: list[torch.Tensor],
    sae: SAE,
    device: torch.device,
    batch: int = 100,
):
    activations_size = [
        activations.shape[1] for activations in activations_list
    ]  # [a, b, ...]
    #print(len(activations_list))
    activations_list = torch.cat(activations_list, dim=1)  # tensor(1, a+b+..., 2048)
    #print(activations_list.shape)
    top_acts = []
    top_indices = []
    # list[tensor(1, batch, 2048), ...]
    chunks = torch.split(activations_list, batch, dim=1)
    
    #split_start_idx = 0
    #for seq_len in activations_size:
    #    chunk = activations_list[:, split_start_idx:split_start_idx + seq_len, :]
    #    split_start_idx += seq_len
    for chunk in chunks:
        #sae_features = sae.encode(chunk.squeeze(0).to(device))
        #top_acts.append(sae_features.top_acts.unsqueeze(0).cpu())
        #top_indices.append(sae_features.top_indices.unsqueeze(0).cpu())
        sae.eval()
        input_activation = chunk.squeeze(0).to(device)
        with torch.no_grad():
            feature_acts = sae.encode(input_activation) 
            
            K = chunk.shape[1]
            flat_acts = feature_acts.flatten(start_dim=0)
            top_values, top_indices_batch = torch.topk(flat_acts, k=K, dim=-1)
        top_acts.append(top_values.unsqueeze(0).cpu())
        top_indices.append(top_indices_batch.unsqueeze(0).cpu())
        #print(top_values.shape)
    
    top_acts = torch.cat(top_acts, dim=1)
    #print(top_acts.shape)
    top_acts = torch.split(top_acts, activations_size, dim=1)

    top_indices = torch.cat(top_indices, dim=1)
    top_indices = torch.split(top_indices, activations_size, dim=1)

    all_sae_features = []

    for top_acts, top_indices in zip(top_acts, top_indices):
        all_sae_features.append(EncoderOutput(top_acts, top_indices, None))

    return all_sae_features


class DatasetConfig(TypedDict):
    name: str
    lang: str
    split: str
    start: int
    end: int | float


def collect_all_activations(
    llm: LanguageModel,
    layers: list[str],
    dataset: Dataset,
    prompt_template: str,
) -> dict[str, list[torch.Tensor]]:
    all_activations = {layer: [] for layer in layers}

    for row in tqdm(dataset, desc="Processing Samples", leave=False):
        prompt = prompt_template.format_map(row)
        activations = collect_activations(llm, layers, prompt)

        for layer, layer_activations in activations.items():
            all_activations[layer].append(layer_activations)

    return all_activations


def collect_all_sae_features(
    all_activations: dict[str, list[torch.Tensor]],
    layers: list[str],
    model: str,
    sae_model: str,
    local_sae_dir: Path,
    batch: int,
):
    all_sae_features = {layer: [] for layer in layers}

    device = get_device()

    for layer, layer_activations in all_activations.items():
        sae = load_sae(model, sae_model, layer, local_sae_dir).to(device)
        all_sae_features[layer] = sae_features_from_activations(
            layer_activations, sae, device, batch
        )

    return all_sae_features
