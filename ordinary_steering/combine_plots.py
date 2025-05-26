#!/usr/bin/env python
"""
Combine plots into single figures with subplots.

This script:
1. Finds all plot images in the ordinary_steering_logs directory
2. Groups them by directories or metrics types
3. Combines related images into subplots
4. Saves the combined images
"""

import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def get_plot_files(plot_dir):
    """Get all plot files from a directory and its subdirectories."""
    plot_files = []
    for ext in ["png", "jpg", "jpeg"]:
        plot_files.extend(
            glob.glob(os.path.join(plot_dir, f"**/*.{ext}"), recursive=True)
        )
    return plot_files


def group_plots_by_directory(plot_files):
    """Group plot files by their immediate parent directory."""
    directory_groups = {}

    for plot_file in plot_files:
        parent_dir = os.path.basename(os.path.dirname(plot_file))
        if parent_dir not in directory_groups:
            directory_groups[parent_dir] = []
        directory_groups[parent_dir].append(plot_file)

    return directory_groups


def group_plots_by_metric(plot_files):
    """Group plot files by metric name in the filename."""
    metric_groups = {}
    metric_patterns = [
        "accuracy",
        "silhouette",
        "auc",
        "class_separability",
        "performance",
        "decision_boundary",
        "vectors",
    ]

    for plot_file in plot_files:
        filename = os.path.basename(plot_file)
        assigned = False

        for metric in metric_patterns:
            if metric in filename.lower():
                if metric not in metric_groups:
                    metric_groups[metric] = []
                metric_groups[metric].append(plot_file)
                assigned = True
                break

        if not assigned:
            if "other" not in metric_groups:
                metric_groups["other"] = []
            metric_groups["other"].append(plot_file)

    return metric_groups


def group_plots_by_strategy(plot_files):
    """Group plot files by embedding strategy."""
    strategy_groups = {}
    strategies = ["last-token", "first-token", "mean", "combined"]

    for plot_file in plot_files:
        path = plot_file.lower()
        assigned = False

        for strategy in strategies:
            if strategy in path:
                if strategy not in strategy_groups:
                    strategy_groups[strategy] = []
                strategy_groups[strategy].append(plot_file)
                assigned = True
                break

        if not assigned:
            if "other" not in strategy_groups:
                strategy_groups["other"] = []
            strategy_groups["other"].append(plot_file)

    return strategy_groups


def group_plots_by_layer(plot_files):
    """Group plot files by layer number."""
    layer_groups = {}
    layer_pattern = re.compile(r"layer[_-](\d+)")

    for plot_file in plot_files:
        path = plot_file.lower()
        match = layer_pattern.search(path)

        if match:
            layer_num = match.group(1)
            if layer_num not in layer_groups:
                layer_groups[layer_num] = []
            layer_groups[layer_num].append(plot_file)
        else:
            if "other" not in layer_groups:
                layer_groups["other"] = []
            layer_groups["other"].append(plot_file)

    return layer_groups


def process_groups(groups, output_dir, max_per_figure):
    """Process each group and create combined plots."""
    os.makedirs(output_dir, exist_ok=True)

    for group_name, files in groups.items():
        if len(files) <= 1:
            print(f"Skipping group '{group_name}' as it has only {len(files)} files")
            continue

        output_path = os.path.join(output_dir, f"combined_{group_name}")
        _combine_images_into_subplots(
            files, output_path, max_per_figure, title=group_name
        )


def _extract_plot_metadata(img_path):
    """Extract useful metadata from a plot file path to create more informative titles."""
    filename = os.path.basename(img_path)
    parent_dir = os.path.basename(os.path.dirname(img_path))
    grandparent_dir = os.path.basename(os.path.dirname(os.path.dirname(img_path)))

    # Extract metric type
    metric_patterns = {
        "accuracy": "Accuracy",
        "silhouette": "Silhouette Score",
        "auc": "AUC Score",
        "class_separability": "Class Separability",
        "performance": "Performance",
        "decision_boundary": "Decision Boundary",
        "vectors": "Vector Representation",
    }

    metric_type = "Unknown Metric"
    for pattern, label in metric_patterns.items():
        if pattern in filename.lower():
            metric_type = label
            break

    # Extract strategy type
    strategy_patterns = {
        "last-token": "Last Token Strategy",
        "first-token": "First Token Strategy",
        "mean": "Mean Pooling Strategy",
        "combined": "Combined Strategy",
    }

    strategy_type = ""
    for pattern, label in strategy_patterns.items():
        if pattern in img_path.lower():
            strategy_type = label
            break

    # Extract data pair type
    pair_patterns = {
        "hate_yes_to_safe_yes": "Hate Yes → Safe Yes",
        "safe_no_to_hate_no": "Safe No → Hate No",
        "hate_yes_to_hate_no": "Hate Yes → Hate No",
        "safe_yes_to_safe_no": "Safe Yes → Safe No",
        "combined": "Combined Pairs",
    }

    pair_type = ""
    for pattern, label in pair_patterns.items():
        if pattern in img_path.lower():
            pair_type = label
            break

    # Extract steering coefficient information
    coef_match = re.search(r"coef[_-](\d+\.?\d*)", img_path.lower())
    coef_info = f"Coef {coef_match.group(1)}" if coef_match else ""

    # Also look for coefficient in the filename
    if not coef_info:
        coef_match = re.search(r"coef=(\d+\.?\d*)", filename.lower())
        coef_info = f"Coef {coef_match.group(1)}" if coef_match else ""

    # Extract layer information
    layer_match = re.search(r"layer[_-](\d+)", img_path.lower())
    layer_info = f"Layer {layer_match.group(1)}" if layer_match else ""

    # Build a comprehensive title
    title_parts = []
    if metric_type:
        title_parts.append(metric_type)
    if strategy_type:
        title_parts.append(strategy_type)
    if pair_type:
        title_parts.append(pair_type)
    if coef_info:
        title_parts.append(coef_info)
    if layer_info:
        title_parts.append(layer_info)

    # Add directory context if we couldn't extract specific information
    if not strategy_type and parent_dir not in title_parts:
        title_parts.append(f"Dir: {parent_dir}")

    # Build final title
    detailed_title = " | ".join(filter(None, title_parts))
    if not detailed_title:
        detailed_title = filename

    return {
        "filename": filename,
        "detailed_title": detailed_title,
        "metric": metric_type,
        "strategy": strategy_type,
        "pair": pair_type,
        "coefficient": coef_info,
        "layer": layer_info,
        "parent_dir": parent_dir,
    }


def _combine_images_into_subplots(
    image_files, output_path, max_per_figure=9, title=None
):
    """Combine multiple images into a single figure with subplots."""
    n_images = len(image_files)
    if n_images == 0:
        print(f"No images to combine for {title if title else 'group'}")
        return

    n_figures = (n_images + max_per_figure - 1) // max_per_figure

    for fig_idx in range(n_figures):
        start_idx = fig_idx * max_per_figure
        end_idx = min(start_idx + max_per_figure, n_images)
        current_images = image_files[start_idx:end_idx]
        n_current = len(current_images)

        # Determine subplot grid dimensions
        n_cols = min(3, n_current)
        n_rows = (n_current + n_cols - 1) // n_cols

        # Create figure with extra space for summary table
        plt.figure(figsize=(7 * n_cols, 7 * n_rows))
        fig_title = (
            f"{title} (Part {fig_idx+1})"
            if title
            else f"Combined Plots (Part {fig_idx+1})"
        )
        plt.suptitle(fig_title, fontsize=16)

        # Collect metadata for summary table
        all_metadata = []

        # Add each image as a subplot
        for i, img_path in enumerate(current_images):
            ax = plt.subplot(n_rows, n_cols, i + 1)

            # Extract metadata for better title
            metadata = _extract_plot_metadata(img_path)
            all_metadata.append(metadata)

            # Load and display image
            img = Image.open(img_path)
            plt.imshow(np.array(img))

            # Set detailed title and original filename as smaller text
            plt.title(metadata["detailed_title"], fontsize=11, wrap=True)
            plt.annotate(
                f"File: {metadata['filename']}",
                xy=(0.5, -0.05),
                xycoords="axes fraction",
                ha="center",
                fontsize=8,
                color="gray",
            )

            # Add a color-coded legend of the conditions shown
            if (
                metadata["strategy"]
                or metadata["pair"]
                or metadata["layer"]
                or metadata["coefficient"]
            ):
                legend_text = []
                if metadata["strategy"]:
                    legend_text.append(f"Strategy: {metadata['strategy']}")
                if metadata["pair"]:
                    legend_text.append(f"Pair: {metadata['pair']}")
                if metadata["coefficient"]:
                    legend_text.append(f"{metadata['coefficient']}")
                if metadata["layer"]:
                    legend_text.append(f"{metadata['layer']}")

                if legend_text:
                    plt.annotate(
                        "\n".join(legend_text),
                        xy=(0.5, -0.15),
                        xycoords="axes fraction",
                        ha="center",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="aliceblue", alpha=0.7),
                    )

            plt.axis("off")

        # Add summary table
        if all_metadata:
            # Create comparison text to highlight what differs between plots
            key_differences = _identify_key_differences(all_metadata)
            comparison_text = _generate_comparison_text(all_metadata, key_differences)

            # Add the comparison text as a table at the bottom
            plt.figtext(
                0.5,
                0.02,
                comparison_text,
                ha="center",
                va="bottom",
                fontsize=11,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="mintcream",
                    edgecolor="gray",
                    alpha=0.9,
                ),
                wrap=True,
            )

        # Save figure with better quality
        output_file = (
            f"{output_path}_{fig_idx+1}.png" if n_figures > 1 else f"{output_path}.png"
        )
        plt.tight_layout(rect=(0, 0.1, 1, 0.95))  # Make room for the title and summary
        plt.savefig(output_file, dpi=200, bbox_inches="tight")
        plt.close()

        # print(f"Saved combined figure to {output_file}")


def _identify_key_differences(metadata_list):
    """Identify which attributes differ across the metadata list."""
    if not metadata_list:
        return []

    key_differences = []
    attributes = ["metric", "strategy", "pair", "coefficient", "layer", "parent_dir"]

    for attr in attributes:
        values = [meta[attr] for meta in metadata_list if meta[attr]]
        unique_values = set(values)

        if len(unique_values) > 1:
            key_differences.append(attr)

    return key_differences


def _generate_comparison_text(metadata_list, key_differences):
    """Generate a comparison text highlighting the key differences."""
    if not metadata_list or not key_differences:
        return "Plots show similar conditions."

    # Create a mapping for readable attribute names
    attr_names = {
        "metric": "Metric",
        "strategy": "Embedding Strategy",
        "pair": "Data Pair Type",
        "coefficient": "Steering Coefficient",
        "layer": "Layer",
        "parent_dir": "Directory",
    }

    n_plots = len(metadata_list)

    # Start with an introduction
    if len(key_differences) == 1:
        diff_attr = attr_names[key_differences[0]]
        text = f"These {n_plots} plots differ by {diff_attr}:\n\n"
    else:
        diff_attrs = ", ".join([attr_names[d] for d in key_differences])
        text = f"These {n_plots} plots differ by {diff_attrs}:\n\n"

    # Create a compact table-like text
    rows = []

    # Add header
    header = ["Plot"]
    for diff in key_differences:
        header.append(attr_names[diff])
    rows.append(" | ".join(header))
    rows.append("-" * (sum(len(h) for h in header) + 3 * len(header)))

    # Add each plot's information
    for i, meta in enumerate(metadata_list):
        row = [f"#{i+1}"]
        for diff in key_differences:
            value = meta[diff] if meta[diff] else "N/A"
            row.append(value)
        rows.append(" | ".join(row))

    # Add common properties if any
    common_props = []
    for attr in ["metric", "strategy", "pair", "coefficient", "layer"]:
        if attr not in key_differences:
            values = [meta[attr] for meta in metadata_list if meta[attr]]
            if values and all(v == values[0] for v in values):
                common_props.append(f"{attr_names[attr]}: {values[0]}")

    if common_props:
        rows.append("")
        rows.append("Common properties:")
        rows.extend([f"• {prop}" for prop in common_props])

    return "\n".join(rows)
