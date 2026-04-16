from typing import Tuple


def pattern_semantics(pattern: Tuple[str, ...]) -> str:
    readable = []
    for item in pattern:
        feature, state = item.split("=", 1)
        display_feature = feature
        if feature.endswith("_prev_mean"):
            display_feature = f"mean {feature[:-10]}"
        elif feature.endswith("_prev_max"):
            display_feature = f"max {feature[:-9]}"
        elif feature.endswith("_prev_last"):
            display_feature = f"last {feature[:-10]}"
        elif feature.endswith("_prev_sum"):
            display_feature = f"total {feature[:-9]}"
        elif feature.endswith("_prev_min"):
            display_feature = f"min {feature[:-9]}"
        readable.append(f"{state.replace('_', ' ')} {display_feature}")
    return " + ".join(readable)


def pattern_name(pattern: Tuple[str, ...]) -> str:
    return "__".join(item.replace("=", "_") for item in pattern)
