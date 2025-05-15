import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import umap
from const import (
    lang_choices_to_iso639_1,
    lang_choices_to_qualified_name,
    language_colors,
    layer_to_index,
)
from loader import load_sae
from utils import get_project_dir


def get_layer_to_lang_count(
    layer_to_statistics: dict[str, pd.DataFrame], layers: list[str], config: dict
) -> dict[str, pd.DataFrame]:
    layer_to_lang_count = {}

    for layer in layers:
        df_feature_index_lang = layer_to_statistics[layer][["index", "lang"]]
        df_feaure_index_lang_count = (
            df_feature_index_lang.groupby("index").count().reset_index()
        )

        df_lang_feature_count = (
            df_feaure_index_lang_count.groupby("lang").count().reset_index()
        )
        df_lang_feature_count.rename(
            columns={"lang": "lang_count", "index": "feature_count"}, inplace=True
        )

        num_latents = config["sae"]["num_latents"]
        total_feature_count = df_lang_feature_count["feature_count"].sum()
        total_nonactivated_feature = num_latents - total_feature_count

        num_feature_count = len(df_lang_feature_count)
        df_lang_feature_count.loc[num_feature_count] = [0, total_nonactivated_feature]
        df_lang_feature_count.sort_values(by="lang_count", inplace=True)

        layer_to_lang_count[layer] = df_lang_feature_count

    return layer_to_lang_count


def combine_layer_to_lang_count(layer_to_lang_count: dict[str, pd.DataFrame]):
    combined_df = None

    for layer, df_lang_feature_count in layer_to_lang_count.items():
        df_lang_feature_count["layer"] = layer

        combined_df = pd.concat([combined_df, df_lang_feature_count], ignore_index=True)

    return combined_df


def plot_layer(
    layer_to_statistics: dict[str, pd.DataFrame], title: str, output_path: Path
):
    fig = px.scatter(
        layer_to_statistics,
        x="index",
        y="count",
        color="lang",
        hover_data=[
            "index",
            "lang",
            "count",
            "avg",
            "q1",
            "median",
            "q3",
            "min_active",
            "max_active",
            "std",
        ],
        title=title,
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_all_layers(layer_to_statistics: dict[str, pd.DataFrame], config: dict):
    for layer in config["layers"]:
        layer_to_statistics = layer_to_statistics[layer]

        model_name = model_name
        sae_model_name = sae_model_name
        dataset_name = dataset_name

        title = f"SAE Features for {layer}\n({model_name} - {sae_model_name} - {dataset_name})"
        output_path = (
            get_project_dir()
            / "visualization"
            / "plot"
            / model_name
            / sae_model_name
            / dataset_name
            / f"{layer}.html"
        )

        plot_layer(layer_to_statistics, title, output_path)


def plot_lang_feature_overlap(
    df_lang_feature_overlap_layer: pd.DataFrame,
    title: str,
    output_path: Path,
    range_y: tuple[int, int],
):
    fig = px.bar(
        df_lang_feature_overlap_layer,
        x="lang_count",
        y="feature_count",
        color="lang_count",
        title=title,
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
        ),
    )

    fig.update_yaxes(range=range_y)

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_combined_lang_feature_overlap(
    df_lang_feature_overlap_layer: pd.DataFrame, title: str, output_path: Path
):
    combined_df = combine_layer_to_lang_count(df_lang_feature_overlap_layer)

    fig = px.bar(
        combined_df, x="lang_count", y="feature_count", color="layer", title=title
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
        ),
    )

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_all_lang_feature_overlap(
    layer_to_statistics: dict[str, pd.DataFrame], config: dict, range_y: tuple[int, int]
):
    df_lang_feature_count_layers = get_layer_to_lang_count(
        layer_to_statistics, config["layers"], config
    )

    model_name = model_name
    sae_model_name = sae_model_name
    dataset_name = dataset_name

    for layer in config["layers"]:
        df_lang_feature_count_layer = df_lang_feature_count_layers[layer]
        title = title = (
            f"SAE Features Overlap for {layer}\n({model_name} - {sae_model_name} - {dataset_name})"
        )
        output_path = (
            get_project_dir()
            / "visualization"
            / "overlap"
            / model_name
            / sae_model_name
            / dataset_name
            / f"{layer}.html"
        )

        plot_lang_feature_overlap(
            df_lang_feature_count_layer, title, output_path, range_y
        )

    title = title = (
        f"SAE Features Overlap \n({model_name} - {sae_model_name} - {dataset_name})"
    )
    output_path = (
        get_project_dir()
        / "visualization"
        / "overlap"
        / model_name
        / sae_model_name
        / dataset_name
        / f"combined.html"
    )
    plot_combined_lang_feature_overlap(df_lang_feature_count_layers, title, output_path)


def plot_lang_feature_overlap_trend(
    layer_to_statistics: dict[str, pd.DataFrame], config: dict
):
    df_lang_feature_count_layers = get_layer_to_lang_count(
        layer_to_statistics, config["layers"], config
    )

    combined_df = combine_layer_to_lang_count(df_lang_feature_count_layers)

    model_name = config["model"].split("/")[-1]
    sae_model_name = config["sae"]["model"].split("/")[-1]
    dataset_name = config["dataset"].split("/")[-1]

    title = f"SAE Features Overlap\n({model_name} - {dataset_name})"

    fig = px.line(
        combined_df,
        x="layer",
        y="feature_count",
        color="lang_count",
        title=title,
        color_discrete_map=language_colors,
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
        ),
    )

    output_path = (
        get_project_dir()
        / "visualization"
        / "trend"
        / model_name
        / sae_model_name
        / dataset_name
        / "sae_features_overlap_trend.html"
    )

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_heatmap(
    df: pd.DataFrame, title: str, output_path: Path, labels: dict[str, str]
):
    fig = px.imshow(
        df,
        labels=labels,
        x=df.columns,
        y=df.index,
        color_continuous_scale="Blues",
        title=title,
        text_auto=True,
    )

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def get_lang_to_feature_indexex(df: pd.DataFrame):
    return df.groupby("lang")["index"].apply(set)


def get_feature_index_to_langs(df: pd.DataFrame):
    return df.groupby("index")["lang"].apply(set)


def get_lang_specific_feature_index(df: pd.DataFrame, specific_feature_lang_count: int):
    feature_index_to_langs = get_feature_index_to_langs(df)
    specific_feature_index_to_langs = feature_index_to_langs[
        feature_index_to_langs.apply(len) == specific_feature_lang_count
    ]
    langs = df["lang"].unique()

    lang_specific_feature_index = {lang: {} for lang in langs}

    for lang in langs:
        lang_specific_feature_index[lang] = set(
            specific_feature_index_to_langs[
                specific_feature_index_to_langs.apply(
                    lambda feature_langs: lang in feature_langs
                )
            ].index
        )

    return pd.Series(lang_specific_feature_index)


def plot_cross_co_occurrence(
    config1: dict,
    df1: pd.DataFrame,
    config2: dict,
    df2: pd.DataFrame,
    title: str,
    output_path: Path,
    specific_feature_lang_count: int | None = None,
):
    lang_to_indices1 = (
        get_lang_to_feature_indexex(df1)
        if specific_feature_lang_count is None
        else get_lang_specific_feature_index(df1, specific_feature_lang_count)
    )
    lang_to_indices2 = (
        get_lang_to_feature_indexex(df2)
        if specific_feature_lang_count is None
        else get_lang_specific_feature_index(df2, specific_feature_lang_count)
    )

    langs1 = config1["languages"]
    langs2 = config2["languages"]

    df_co_occurrence_matrix = pd.DataFrame(
        index=langs1, columns=langs2, dtype=int
    ).fillna(0)

    for lang1 in langs1:
        for lang2 in langs2:
            df_co_occurrence_matrix.loc[lang1, lang2] = len(
                lang_to_indices1[lang1] & lang_to_indices2[lang2]
            )

    dataset1_name = config1["dataset"]
    dataset2_name = config2["dataset"]

    fig = px.imshow(
        df_co_occurrence_matrix,
        labels={
            "y": f"Language ({dataset1_name})",
            "x": f"Language ({dataset2_name})",
            "color": "Shared Indices",
        },
        x=df_co_occurrence_matrix.columns,
        y=df_co_occurrence_matrix.index,
        color_continuous_scale="Blues",
        title=title,
        text_auto=True,
    )

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)

    return df_co_occurrence_matrix


def plot_all_cross_co_occurrence(
    layer_to_statistics1: dict[str, pd.DataFrame],
    config1: dict,
    layer_to_statistics2: dict[str, pd.DataFrame],
    config2: dict,
    specific_feature_lang_count: int | None = None,
):
    overlap_layers = list(filter(lambda x: x in config2["layers"], config1["layers"]))

    df_combined_co_occurrence_matrix = None

    labels = {
        "y": f"Language ({dataset1_name})",
        "x": f"Language ({dataset2_name})",
        "color": "Shared Indices",
    }

    folder_name = (
        "all"
        if specific_feature_lang_count is None
        else f"count-{specific_feature_lang_count}"
    )
    lang_count_info = (
        ""
        if specific_feature_lang_count is None
        else f"Lang Count {specific_feature_lang_count} "
    )

    for layer in overlap_layers:
        layer_to_statistics1 = layer_to_statistics1[layer]
        layer_to_statistics2 = layer_to_statistics2[layer]

        model1_name = model1_name
        sae_model1_name = sae_model1_name
        dataset1_name = dataset1_name

        model2_name = model2_name
        sae_model2_name = sae_model2_name
        dataset2_name = dataset2_name

        title = f"SAE Features Cross Co-Occurrence {lang_count_info}for {layer} ({model1_name} - {sae_model1_name} - {dataset1_name} and {dataset2_name})"

        output_path = (
            get_project_dir()
            / "visualization"
            / "cross-co-occurrence"
            / model1_name
            / sae_model1_name
            / dataset1_name
            / dataset2_name
            / folder_name
            / f"{layer}.html"
        )

        df_co_occurrence_matrix = plot_cross_co_occurrence(
            config1,
            layer_to_statistics1,
            config2,
            layer_to_statistics2,
            title,
            output_path,
            specific_feature_lang_count,
        )

        df_combined_co_occurrence_matrix = (
            df_combined_co_occurrence_matrix + df_co_occurrence_matrix
            if df_combined_co_occurrence_matrix is not None
            else df_co_occurrence_matrix
        )

    title = f"SAE Features Cross Co-Occurrence {lang_count_info} ({model1_name} - {sae_model1_name} - {dataset1_name} and {dataset2_name})"
    output_path = (
        get_project_dir()
        / "visualization"
        / "cross-co-occurrence"
        / model1_name
        / sae_model1_name
        / dataset1_name
        / dataset2_name
        / folder_name
        / f"combined.html"
    )
    plot_heatmap(df_combined_co_occurrence_matrix, title, output_path, labels=labels)


def plot_all_co_occurrence(
    layer_to_statistics: dict[str, pd.DataFrame],
    config: dict,
    specific_feature_lang_count: int | None = None,
):

    df_combined_co_occurrence_matrix = None

    folder_name = (
        "all"
        if specific_feature_lang_count is None
        else f"count-{specific_feature_lang_count}"
    )
    lang_count_info = (
        ""
        if specific_feature_lang_count is None
        else f"Lang Count {specific_feature_lang_count} "
    )

    model_name = config["model"].split("/")[-1]
    sae_model_name = config["sae"]["model"].split("/")[-1]
    dataset_name = config["dataset"].split("/")[-1]

    for layer in config["layers"]:
        layer_to_statistics = layer_to_statistics[layer]
        title = title = (
            f"SAE Features Co-Occurrence {lang_count_info}for {layer}\n({model_name} _ {sae_model_name} - {dataset_name})"
        )
        output_path = (
            get_project_dir()
            / "visualization"
            / "co-occurrence"
            / model_name
            / sae_model_name
            / dataset_name
            / folder_name
            / f"{layer}.html"
        )

        df_co_occurrence_matrix = plot_cross_co_occurrence(
            config,
            layer_to_statistics,
            config,
            layer_to_statistics,
            title,
            output_path,
            specific_feature_lang_count,
        )

        df_combined_co_occurrence_matrix = (
            df_combined_co_occurrence_matrix + df_co_occurrence_matrix
            if df_combined_co_occurrence_matrix is not None
            else df_co_occurrence_matrix
        )

    title = title = (
        f"SAE Features Co-Occurrence\n({model_name} - {sae_model_name} - {dataset_name})"
    )
    output_path = (
        get_project_dir()
        / "visualization"
        / "co-occurrence"
        / model_name
        / sae_model_name
        / dataset_name
        / folder_name
        / f"combined.html"
    )
    plot_heatmap(
        df_combined_co_occurrence_matrix,
        title,
        output_path,
        labels=dict(x=f"Language", y=f"Language", color="Shared Indices"),
    )


def plot_box_plot(
    layer_to_statistics: dict[str, pd.DataFrame],
    layers: list[str],
    lang: str,
    x: str,
    y: str,
    title: str,
    output_path: Path,
):
    fig = go.Figure()

    for i, layer in enumerate(layers):
        layer_to_statistics_lang = layer_to_statistics[layer][
            layer_to_statistics[layer]["lang"] == lang
        ]

        data = layer_to_statistics_lang[y]

        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        upper_bound = q3 + 1.5 * iqr

        # Find outliers
        outliers = data[data > upper_bound]

        fig.add_trace(
            go.Box(
                x=layer_to_statistics_lang[x],
                y=data,
                name=layer,
                boxpoints=False,
            )
        )

        ranges = [
            (upper_bound, 1000, "mild"),
            (1000, 10000, "moderate"),
            (10000, float("inf"), "extreme"),
        ]

        number_of_count = len(layer_to_statistics_lang["count"])
        annotation_text = [
            f"{number_of_count} total active features",
        ]
        for lower, upper, label in ranges:
            range_outliers = outliers[(outliers > lower) & (outliers <= upper)]
            if len(range_outliers) > 0:
                annotation_text.append(
                    f"{len(range_outliers)} {label} outliers ({lower:.0f}-{upper if upper != float('inf') else 'inf'}) max={range_outliers.max():.0f}"
                )

        if annotation_text:
            fig.add_annotation(
                x=i,
                y=10,
                text="<br>".join(annotation_text),
                showarrow=False,
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                opacity=0.8,
            )

    fig.update_layout(
        title=title,
        yaxis_type="log",
    )

    os.makedirs(output_path.parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_all_count_box_plots(
    layer_to_statistics: dict[str, pd.DataFrame],
    config: dict,
):
    model_name = config["model"].split("/")[-1]
    sae_model_name = config["sae"]["model"].split("/")[-1]
    dataset_name = config["dataset"].split("/")[-1]

    for lang in config["languages"]:
        title = title = (
            f"SAE Features Count Box Plot for {lang}\n({model_name} - {sae_model_name} - {dataset_name})"
        )
        output_path = (
            get_project_dir()
            / "visualization"
            / "count-box-plot"
            / model_name
            / sae_model_name
            / dataset_name
            / f"{lang}.html"
        )

        plot_box_plot(
            layer_to_statistics,
            config["layers"],
            lang,
            "layer",
            "count",
            title,
            output_path,
        )


def plot_lape_result(lape_result: dict, title, out_dir: Path):
    sorted_lang = lape_result["sorted_lang"]

    df_all_langs = None

    for lang_index, lang in enumerate(sorted_lang):
        final_indice = lape_result["final_indice"][lang_index]
        frequency_count = list(
            map(lambda layer_index: layer_index.shape[0], final_indice)
        )
        indices = list(range(len(frequency_count)))

        df_lang = pd.DataFrame(
            {
                "Layer": indices,
                "Count": frequency_count,
                "Lang": lang_choices_to_iso639_1[lang],
            }
        )

        plot_specific_lape_lang(lang, df_lang, title, out_dir)

        df_all_langs = pd.concat([df_all_langs, df_lang], ignore_index=True)

    plot_combined_lape_lang(df_all_langs, title, out_dir)
    plot_combined_lape_lang_count(df_all_langs, title, out_dir)


def plot_specific_lape_lang(lang, df_lang, title, out_dir):
    fig = px.bar(
        df_lang,
        x="Layer",
        y="Count",
        title=f"{title} ({lang})",
        color="Lang",
        labels={"Layer": "Layer", "Count": "Count"},
        color_discrete_map=language_colors,
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
        )
    )

    fig.update_layout(plot_bgcolor="white")

    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )

    fig.update_layout(showlegend=True)

    output_dir = get_project_dir() / out_dir
    output_path = output_dir / f"lape_{lang}.html"

    os.makedirs(output_dir, exist_ok=True)

    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_combined_lape_lang(df_all_langs, title, out_dir):
    fig = px.bar(
        df_all_langs,
        x="Layer",
        y="Count",
        color="Lang",
        title=title,
        labels={"Layer": "Layer", "Count": "Count"},
        color_discrete_map=language_colors,
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
        )
    )

    fig.update_layout(plot_bgcolor="white")

    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )

    output_dir = get_project_dir() / out_dir
    output_path = output_dir / f"lape.html"

    os.makedirs(output_dir, exist_ok=True)

    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_combined_lape_lang_count(df_all_langs, title, out_dir):
    df_copy = df_all_langs.copy()

    df_copy["Layer"] = df_copy["Layer"].astype(str)

    # Create a color map for layers 0-15
    layer_colors = {
        "0": "#1f77b4",  # blue
        "1": "#ff7f0e",  # orange
        "2": "#2ca02c",  # green
        "3": "#d62728",  # red
        "4": "#9467bd",  # purple
        "5": "#8c564b",  # brown
        "6": "#e377c2",  # pink
        "7": "#7f7f7f",  # gray
        "8": "#bcbd22",  # olive
        "9": "#17becf",  # teal
        "10": "#aec7e8", # light blue
        "11": "#ffbb78", # light orange
        "12": "#98df8a", # light green
        "13": "#ff9896", # light red
        "14": "#c5b0d5", # light purple
        "15": "#c49c94"  # light brown
    }
    
    # Group by Lang and sum Counts to determine sorting order
    lang_totals = df_copy.groupby("Lang")["Count"].sum().reset_index()
    lang_totals = lang_totals.sort_values("Count", ascending=True)
    sorted_langs = lang_totals["Lang"].tolist()
    
    # Create a categorical column with custom order
    df_copy["Lang"] = pd.Categorical(df_copy["Lang"], categories=sorted_langs, ordered=True)
    
    # Sort the dataframe
    df_copy = df_copy.sort_values("Lang")
    
    # Create figure with sorted x-axis
    fig = px.bar(
        df_copy,
        x="Lang",
        y="Count",
        color="Layer",
        title=title,
        color_discrete_map=layer_colors,
        category_orders={"Layer": [str(i) for i in range(16)]}  # Ensure layers are ordered from 0-15
    )

    fig.update_layout(
        xaxis=dict(
            tickmode="linear",
        )
    )

    fig.update_layout(plot_bgcolor="white")

    fig.update_xaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
    )

    fig.update_yaxes(
        mirror=True,
        ticks="outside",
        showline=True,
        linecolor="black",
        gridcolor="lightgrey",
    )

    output_dir = get_project_dir() / out_dir
    output_path = output_dir / f"feature_counts.html"

    os.makedirs(output_dir, exist_ok=True)

    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def plot_umap(
    lape_result,
    layers: list[str],
    model_name: str,
    sae_model_name: str,
    output_dir: Path,
    interpretations: dict,
    seed: int = 42,
    working_dir: Path | None = None,
):
    for layer in layers:
        layer_index = layer_to_index[layer]
        sae = load_sae(model_name, sae_model_name, layer)

        all_features = [*range(sae.W_dec.size(0))]
        layer_features = []
        layer_feature_indices = []
        layer_feature_langs = []
        layer_feature_interpretations = []
        layer_feature_entropies = []
        layer_feature_selected_probs = []

        for lang_final_indices, lang_sae_features, lang in zip(
            lape_result["final_indice"],
            lape_result["sae_features"],
            lape_result["sorted_lang"],
        ):
            layer_lang_sae_features = lang_sae_features[layer_index].tolist()
            layer_lang_feature_indices = lang_final_indices[layer_index].tolist()
            layer_lang_feature_langs = [lang_choices_to_iso639_1[lang]] * len(
                layer_lang_sae_features
            )
            layer_lang_feature_interpretations = [
                "<br>".join(
                    textwrap.wrap(interpretations[layer][feature_index], width=40)
                )
                for feature_index in layer_lang_feature_indices
            ]

            selected_probs = lape_result["features_info"][lang]["selected_probs"]
            entropies = lape_result["features_info"][lang]["entropies"]

            layer_lang_feature_entropies = []
            layer_lang_feature_selected_probs = []

            for feature_index in layer_lang_feature_indices:
                arg_index = lape_result["features_info"][lang]["indicies"].index(
                    (layer_index, feature_index)
                )
                entropy = round(entropies[arg_index].item(), ndigits=3)
                selected_prob = round(selected_probs[arg_index].item(), ndigits=3)

                layer_lang_feature_entropies.append(entropy)
                layer_lang_feature_selected_probs.append(selected_prob)

            layer_features.extend(layer_lang_sae_features)
            layer_feature_indices.extend(layer_lang_feature_indices)
            layer_feature_langs.extend(layer_lang_feature_langs)
            layer_feature_interpretations.extend(layer_lang_feature_interpretations)
            layer_feature_entropies.extend(layer_lang_feature_entropies)
            layer_feature_selected_probs.extend(layer_lang_feature_selected_probs)

        # https://github.com/lmcinnes/umap/issues/201
        n_components = 2

        if len(layer_features) <= n_components + 1:
            continue

        reducer = umap.UMAP(
            n_neighbors=15,
            metric="cosine",
            min_dist=0.05,
            n_components=n_components,
            random_state=seed,
        )

        remainding_feature_indices = sorted(
            set(all_features) - set(layer_feature_indices)
        )
        remainding_features = sae.W_dec[remainding_feature_indices].detach().tolist()

        layer_features.extend(remainding_features)
        layer_feature_indices.extend(remainding_feature_indices)
        layer_feature_langs.extend((["other"] * len(remainding_features)))
        layer_feature_interpretations.extend((["-"] * len(remainding_features)))
        layer_feature_entropies.extend((["-"] * len(remainding_features)))
        layer_feature_selected_probs.extend((["-"] * len(remainding_features)))

        embedding = reducer.fit_transform(layer_features)

        # Create a custom color palette that assigns grey to "other" and uses plotly colors for languages
        unique_langs = set(layer_feature_langs)
        color_discrete_map = {}

        # Set "other" to grey
        if "other" in unique_langs:
            color_discrete_map["other"] = "rgba(211, 211, 211, 0.5)"

        fig = px.scatter(
            embedding,
            x=0,
            y=1,
            color=layer_feature_langs,
            title=f"UMAP projection of {model_name} - {sae_model_name} - {layer}",
            hover_data={
                "Feature Index": layer_feature_indices,
                "Entropy": layer_feature_entropies,
                "Selected Prob": layer_feature_selected_probs,
                "Interpretation": layer_feature_interpretations,
            },
            color_discrete_map={**language_colors, **color_discrete_map},
        )

        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")

        output_path = output_dir / layer / "umap.html"

        os.makedirs(output_path.parent, exist_ok=True)

        fig.write_html(
            output_path,
            include_plotlyjs="cdn",
        )

        fig = px.scatter(
            embedding,
            x=0,
            y=1,
            color=layer_feature_langs,
            title=f"UMAP projection of {model_name} - {sae_model_name} - {layer}",
            hover_data={
                "Feature Index": layer_feature_indices,
                "Entropy": layer_feature_entropies,
                "Selected Prob": layer_feature_selected_probs,
                "Interpretation": layer_feature_interpretations,
            },
            color_discrete_map={
                **language_colors,
                **{"other": "rgba(211, 211, 211, 0.02)"},
            },
        )

        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")

        save_image(output_path, fig, working_dir)


def plot_ppl_change_matrix(
    langs,
    normal_ppl_result,
    intervened_ppl_results,
    output_path: Path,
    title: str = "PPL Change Matrix",
    num_examples: int | None = None,
):
    num_langs = len(langs)
    ppl_matrix = torch.zeros(num_langs, num_langs)

    for intervened_lang_index, intervened_lang in enumerate(langs):
        normalized_intervened_lang = lang_choices_to_qualified_name[intervened_lang]
        intervened_ppl_result = intervened_ppl_results[normalized_intervened_lang]

        for lang_index, lang in enumerate(langs):
            normalized_lang = lang_choices_to_qualified_name[lang]
            intervened_ppls = np.array(
                intervened_ppl_result[normalized_lang]["perplexities"][:num_examples]
            )
            normal_ppls = np.array(
                normal_ppl_result[normalized_lang]["perplexities"][:num_examples]
            )
            avg_diff = np.mean(intervened_ppls - normal_ppls).round(decimals=2).item()
            ppl_matrix[intervened_lang_index][lang_index] = avg_diff

    iso_langs = [lang_choices_to_iso639_1[lang] for lang in langs]

    fig = px.imshow(
        ppl_matrix.numpy(),
        x=iso_langs,
        y=iso_langs,
        labels=dict(x="Impacted Language", y="Intervened Language", color="Value"),
        text_auto=True,
        color_continuous_scale=["white", "orange"],
        zmin=0,
        zmax=1000,
        aspect="auto",
    )

    # Add border to diagonal cells
    for i in range(len(iso_langs)):
        fig.add_shape(
            type="rect",
            x0=i - 0.5,
            y0=i - 0.5,
            x1=i + 0.5,
            y1=i + 0.5,
            line=dict(color="lightgray", width=1),
        )

    fig.update_layout(xaxis_side="top")

    os.makedirs(output_path.parent, exist_ok=True)

    fig.write_html(output_path, include_plotlyjs="cdn")

    save_image(output_path, fig)


def generate_ppl_change_matrix(
    configs: list[str],
    model: str,
    dataset: str,
    langs: list[str],
    input_dir: Path,
    normal_ppl_result,
):
    iso_lang = [lang_choices_to_iso639_1[lang] for lang in langs]

    for config in configs:
        in_path = input_dir / config

        intervened_sae_features_ppl_results = {
            lang_choices_to_qualified_name[intervened_lang]: torch.load(
                in_path / f"ppl_{intervened_lang}.pt", weights_only=False
            )
            for intervened_lang in langs
        }

        out_path = (
            get_project_dir()
            / "visualization"
            / "ppl"
            / model
            / dataset
            / "sae_intervention"
            / config
            / "ppl_change_matrix.html"
        )

        example_lang = lang_choices_to_qualified_name[langs[0]]
        num_examples = len(
            intervened_sae_features_ppl_results[example_lang][example_lang][
                "perplexities"
            ]
        )

        plot_ppl_change_matrix(
            iso_lang,
            normal_ppl_result,
            intervened_sae_features_ppl_results,
            out_path,
            title=f"PPL Change Matrix for SAE Features Interventions ({num_examples} examples)",
            num_examples=num_examples,
        )


def save_image(path: Path, fig, working_dir: Path | None = None):
    project_dir = get_project_dir() if working_dir is None else working_dir

    relative_path = path.relative_to(project_dir)

    output_path = project_dir / "images" / relative_path
    output_path = output_path.with_suffix(".pdf")

    os.makedirs(output_path.parent, exist_ok=True)

    fig.update_layout(title=None)

    fig.write_image(output_path)
