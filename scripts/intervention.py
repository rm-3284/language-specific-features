from typing import TypedDict

import torch


def sae_features_intervene_last_token(
    hidden_state: torch.Tensor,
    lape: dict[str, any],
    layer_index: int,
    lang_index: int,
    multiplier: float,
    lape_value_type: str = "final_indice_global_max_active",
):
    sae_features = lape["sae_features"][lang_index][layer_index]

    if sae_features.numel() == 0:
        return hidden_state

    max_activations = lape[lape_value_type][lang_index][layer_index].unsqueeze(1)
    steering_vector = torch.sum((multiplier * max_activations) * sae_features, dim=0)
    hidden_state[0, -1] += steering_vector

    return hidden_state


def sae_features_intervene_all_token(
    hidden_state: torch.Tensor,
    lape: dict[str, any],
    layer_index: int,
    lang_index: int,
    multiplier: float,
    lape_value_type: str = "final_indice_global_max_active",
):
    sae_features = lape["sae_features"][lang_index][layer_index]

    if sae_features.numel() == 0:
        return hidden_state

    lape_value = lape[lape_value_type][lang_index][layer_index].unsqueeze(1)
    steering_vector = torch.sum((multiplier * lape_value) * sae_features, dim=0)

    hidden_state += steering_vector

    return hidden_state


def neurons_intervene_all_token(
    hidden_state: torch.Tensor,
    lape: dict[str, any],
    layer_index: int,
    intervention_lang_index: int,
    value: float,
):
    neuron_indices = lape["final_indice"][intervention_lang_index][layer_index]
    hidden_state[:, :, neuron_indices] = value

    return hidden_state
