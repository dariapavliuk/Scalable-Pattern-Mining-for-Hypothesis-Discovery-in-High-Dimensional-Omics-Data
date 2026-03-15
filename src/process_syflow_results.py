import pandas as pd
import numpy as np
import os
import pickle
from scipy.stats import entropy
from pathlib import Path
from functools import reduce
import operator
import re
import math
import matplotlib.pyplot as plt
import gprofiler
import seaborn as sns
import itertools
import re, math

def parse_rules(df, rules):
    parsed_rules = {}

    for i, rule in enumerate(rules, start=1):
        predicates = rule.split('∧')
        parsed_predicates = []

        for pred in predicates:
            pred = pred.strip()

            if pred.startswith('¬'):
                var = pred[1:].strip()
                if var not in df.columns:
                    raise KeyError(f"Column not found: '{var}'")
                mask = (df[var] == 0)
                parsed_predicates.append({
                    'var': var,
                    'predicate': f'¬{var}',
                    'quantile_predicate': f'¬{var}',
                    'predicate_mask': mask
                })

            elif '<' not in pred:
                var = pred.strip()
                if var not in df.columns:
                    raise KeyError(f"Column not found: '{var}'")
                mask = (df[var] == 1)
        
                parsed_predicates.append({
                    'var': var,
                    'predicate': var,
                    'quantile_predicate': var,
                    'predicate_mask': mask,
        
                })

            else:
                m = re.match(
                    r'\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*<\s*(.+?)\s*<\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$',
                    pred
                )
                if m:
                    lower_str, var, upper_str = m.groups()
                    var = var.strip()
                    if var not in df.columns:
                        raise KeyError(f"Column not found: '{var}'")
                    lower = float(lower_str)
                    upper = float(upper_str)

                    series = df[var]
                    mask = (series >= lower) & (series <= upper)

                    lower_q = (series < lower).mean() * 100
                    upper_q = (series < upper).mean() * 100
                    quantile_str = f'Q{math.floor(lower_q)} < {var} < Q{math.ceil(upper_q)}'
                    md = get_amd(mask, df, var)

                    parsed_predicates.append({
                        'var': var,
                        'predicate': f'{lower} < {var} < {upper}',
                        'quantile_predicate': quantile_str,
                        'predicate_mask': mask,
                        'md': md
                    })
                else:
                    raise ValueError(f"Could not parse predicate: '{pred}'")

        parsed_rules[i] = parsed_predicates

    return parsed_rules


def get_subgroup_mask(parsed_rule):
    masks = [p["predicate_mask"] for p in parsed_rule if "predicate_mask" in p]
    if not masks:
        return None
    first = masks[0]
    if isinstance(first, pd.Series):
        idx = first.index
        series_masks = [
            (m if isinstance(m, pd.Series) else pd.Series(m, index=idx)).astype(bool).reindex(idx, fill_value=False)
            for m in masks
        ]
    else:
        series_masks = [pd.Series(m).astype(bool) for m in masks]
        idx = series_masks[0].index
        series_masks = [m.reindex(idx, fill_value=False) for m in series_masks]
    return reduce(operator.and_, series_masks)

def get_subgroup_size(subgroup_mask):

    if subgroup_mask is not None:
        abs_size = np.sum(subgroup_mask)
        prop_size = abs_size / len(subgroup_mask)
    else:
        abs_size = 0
        prop_size = 0.0

    return abs_size, prop_size

def get_kl_divergence(subgroup_mask, df, target_gene):
    
    target_array = df.loc[df.index, target_gene].values
    subgroup_targets = target_array[subgroup_mask]

    # Freedman–Diaconis bins computed on full distribution
    try:
        _, bin_edges = np.histogram(target_array, bins='fd')
    except Exception:
        # fallback if not enough data
     _, bin_edges = np.histogram(target_array, bins=10)

    # Histograms
    full_hist, _ = np.histogram(target_array, bins=bin_edges)
    subgroup_hist, _ = np.histogram(subgroup_targets, bins=bin_edges)

    # Smoothing to avoid zeros
    full_hist = np.maximum(full_hist.astype(float), 1e-10)
    subgroup_hist = np.maximum(subgroup_hist.astype(float), 1e-10)

    # Normalize
    full_hist /= full_hist.sum()
    subgroup_hist /= subgroup_hist.sum()

    kl = entropy(subgroup_hist, full_hist)

    return kl

def get_amd(subgroup_mask, df, target_gene):
    y = df[target_gene].to_numpy(dtype=float)
    y_sub = y[subgroup_mask]

    # Handle degenerate cases
    if len(y_sub) == 0 or np.all(np.isnan(y)) or np.all(np.isnan(y_sub)):
        return np.nan

    # Freedman–Diaconis bins
    try:
        _, bin_edges = np.histogram(y, bins='fd')
    except Exception:
        _, bin_edges = np.histogram(y, bins=10)

    # Histogram counts
    hist_all, _ = np.histogram(y, bins=bin_edges)
    hist_sub, _ = np.histogram(y_sub, bins=bin_edges)

    # Convert to probabilities
    p_all = hist_all / np.sum(hist_all)
    p_sub = hist_sub / np.sum(hist_sub)

    # Compute bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Expected values from histograms
    E_all = np.sum(p_all * bin_centers)
    E_sub = np.sum(p_sub * bin_centers)
    
    md = E_sub - E_all
    return float(md)


def process_parsed_rules(parsed_rules, df, target_gene):
    summary_parsed_rules = {}
    subgroup_masks = {}

    for subgroup, predicates in parsed_rules.items():
        rule = ' ∧ '.join(p['predicate'] for p in predicates)
        quantile_rule = ' ∧ '.join(p['quantile_predicate'] for p in predicates)
        n_predicates = len(predicates)
        vars_list = [p['var'] for p in predicates]
        subgroup_mask = get_subgroup_mask(predicates)
        abs_size, prop_size = get_subgroup_size(subgroup_mask)
        kl = get_kl_divergence(subgroup_mask, df, target_gene)
        md = get_amd(subgroup_mask, df, target_gene)
        amd = np.abs(md)
    


        summary_parsed_rules[subgroup] = {
            "rule": rule,
            "quantile_rule": quantile_rule,
            "n_predicates": n_predicates,
            "vars": vars_list,
            "absolute_subgroup_size": abs_size,
            "proportional_subgroup_size": round(prop_size, 3),
            "kl_divergence": round(kl, 3),
            "md": round(md, 3),
            "amd": round(amd, 3)
 
        }
        subgroup_masks[subgroup] = subgroup_mask

    return summary_parsed_rules, subgroup_masks

def filter_predicates(parsed_rules, threshold=0.75):
    filtered_parsed_rules = {}
    for key, predicates in parsed_rules.items():
        filtered_predicates = []
        for var_dict in predicates:
            mask = var_dict['predicate_mask']
            true_ratio = sum(mask) / len(mask)
            
            if true_ratio <= threshold:
                filtered_predicates.append(var_dict)
        
        filtered_parsed_rules[key] = filtered_predicates
    return filtered_parsed_rules

def select_subgroup(summary_parsed_rules, subgroup_masks, target_col, relevance_dict):  
    try:
        value = relevance_dict[target_col]
        if value >= 0:
            key, summary = next(((k, v) for k, v in summary_parsed_rules.items() if v['md'] >= 0), (None, None))
            subgroup_mask = subgroup_masks[key]
        else:
            key, summary = next(((k, v) for k, v in summary_parsed_rules.items() if v['md'] < 0), (None, None))
            subgroup_mask = subgroup_masks[key]

    except Exception as e:
        summary = None
        subgroup_mask = None
        
    return summary, subgroup_mask

def process_syflow_results_wrapper(
    rules,
    data_df,
    target_col,
    relevance_dict,
    threshold
):
    parsed_rules = parse_rules(data_df, rules)
    filtered = filter_predicates(parsed_rules, threshold)
    summary, masks = process_parsed_rules(filtered, data_df, target_col)
    summary, mask = select_subgroup(summary, masks, target_col, relevance_dict)
    return summary, mask


def select_best_combo(results_dict):
    def score(metrics):
        try:
            return (
                float(metrics["proportional_subgroup_size"])
                * float(metrics["kl_divergence"])
            )
        except (TypeError, KeyError, ValueError):
            return None

    valid = []

    for key, value in results_dict.items():
        if value is None:
            continue

        if len(value) < 2:
            continue

        processed = value[1]

        if processed is None:
            continue

        if not isinstance(processed, (list, tuple)) or len(processed) == 0:
            continue

        metrics = processed[0]

        if metrics is None:
            continue

        s = score(metrics)
        if s is None:
            continue

        valid.append((key, s))

    if not valid:
        return None, None

    best_key = max(valid, key=lambda x: x[1])[0]
    return best_key, results_dict[best_key]