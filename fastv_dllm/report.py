"""Aggregation and attribution for paired FastV/Flash experiments."""

import math


def mean(values):
    return sum(values) / len(values) if values else None


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, math.ceil(len(values) * fraction) - 1)]


def summarize(records, methods):
    result = {}
    for method in methods:
        rows = [row[method] for row in records]
        result[method] = dict(
            examples=len(rows),
            accuracy=mean([row["correct"] for row in rows]),
            mean_seconds=mean([row["seconds"] for row in rows]),
            p95_seconds=percentile([row["seconds"] for row in rows], 0.95),
            total_seconds=sum(row["seconds"] for row in rows),
            mean_nfe=mean([row["logical_forwards"] for row in rows]),
            mean_denoise=mean([row["ordinary_denoise"] for row in rows]),
            mean_tokens=mean([row["tokens"] for row in rows]),
            truncation_rate=mean([row["length_capped"] for row in rows]),
            flash_calls=sum(row["backend"]["flash_calls"] for row in rows),
            fallback_calls=sum(row["backend"]["fallback_calls"] for row in rows),
            mean_deep_tokens=mean([
                token for row in rows for token in row.get("deep_tokens", [])
            ]),
        )
    def ratio(baseline, candidate):
        return result[baseline]["total_seconds"] / result[candidate]["total_seconds"]
    result["attribution"] = {}
    if {"sdpa_native", "flash_native"}.issubset(result):
        result["attribution"]["flash_engineering_speedup"] = ratio("sdpa_native", "flash_native")
    if {"flash_native", "flash_fastv"}.issubset(result):
        result["attribution"]["fastv_method_speedup_same_flash"] = ratio("flash_native", "flash_fastv")
        result["attribution"]["fastv_accuracy_delta_same_flash"] = (
            result["flash_fastv"]["accuracy"] - result["flash_native"]["accuracy"]
        )
    if {"sdpa_native", "sdpa_fastv"}.issubset(result):
        result["attribution"]["fastv_method_speedup_same_sdpa"] = ratio("sdpa_native", "sdpa_fastv")
    if {"flash_native", "flash_cache"}.issubset(result):
        result["attribution"]["official_cache_speedup_same_flash"] = ratio("flash_native", "flash_cache")
    return result
