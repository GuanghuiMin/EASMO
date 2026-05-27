"""Stage 04 — SVD on Jacobian-weighted activation vectors saved by
stage 02 (or recomputed on the fly).

Outputs:
  outputs/tables/active_subspace_spectrum.csv
  outputs/figures/fig_active_subspace_spectrum_example.{png,pdf}
  outputs/figures/fig_active_subspace_spectrum_span.{png,pdf}
  outputs/figures/fig_high_vs_low_sensitivity_spectrum.{png,pdf}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from motivation_v6_jacobian.data import (  # noqa: E402
    ensure_outputs, raw_path, table_path, figure_path, read_jsonl,
)
from motivation_v6_jacobian.active_subspace import spectrum  # noqa: E402
from motivation_v6_jacobian.plotting import plot_spectrum  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None,
                    help="Path to active_vectors_layerX.npz (autodetected if omitted)")
    ap.add_argument("--meta", default=str(raw_path("active_vector_metadata.jsonl")))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--n_components", type=int, default=256)
    args = ap.parse_args()

    ensure_outputs()

    # Auto-detect npz
    npz_path = Path(args.npz) if args.npz else None
    if npz_path is None:
        candidates = sorted(Path(raw_path("").as_posix()).parent.glob(
            "active_vectors_layer*.npz"))
        # raw_path("") yields .../outputs/raw/ — fall back to glob there
        if not candidates:
            candidates = sorted(
                Path("/workspace/EASMO/motivation_v6_jacobian/outputs/raw").glob(
                    "active_vectors_layer*.npz"))
        if not candidates:
            print("[04] no active_vectors_layerX.npz found; "
                  "did you run stage 02 with --capture_active?")
            return
        npz_path = candidates[0]
    npz = np.load(npz_path)
    layer_index = int(str(npz_path.stem).split("layer")[-1])
    print(f"[04] loading {npz_path}  layer={layer_index}")

    example = npz["example"].astype(np.float64)
    span = npz["span"].astype(np.float64)
    print(f"      example matrix: {example.shape}, span matrix: {span.shape}")

    meta_rows = read_jsonl(Path(args.meta))
    span_meta = [m for m in meta_rows if m.get("matrix") == "span"]
    span_meta.sort(key=lambda m: m["row"])
    span_sens = np.array(
        [m.get("v4_final_sensitivity") or 0.0 for m in span_meta],
        dtype=np.float64,
    )
    if len(span_sens) != span.shape[0]:
        print(f"      WARN: span meta count {len(span_sens)} != span rows {span.shape[0]}")
        span_sens = np.zeros(span.shape[0])

    threshold = float(np.median(span_sens))
    high_mask = span_sens >= threshold
    low_mask = span_sens < threshold
    print(f"      median v4 sensitivity = {threshold:.3f}  "
          f"high={int(high_mask.sum())}  low={int(low_mask.sum())}")

    matrices = {
        "example": example,
        "span": span,
        "high_v4": span[high_mask] if high_mask.any() else span[:0],
        "low_v4": span[low_mask] if low_mask.any() else span[:0],
    }

    rows = []
    plot_data: Dict[str, tuple] = {}
    for name, M in matrices.items():
        if M.size == 0:
            continue
        S, expl, cum = spectrum(M, n_components=args.n_components)
        plot_data[name] = (S, expl, cum)
        for i, (s, e, c) in enumerate(zip(S, expl, cum)):
            rows.append({
                "matrix": name,
                "layer_index": layer_index,
                "component": i + 1,
                "singular_value": float(s),
                "explained_variance": float(e),
                "cumulative_explained_variance": float(c),
            })

    df = pd.DataFrame(rows)
    out_path = table_path("active_subspace_spectrum.csv")
    df.to_csv(out_path, index=False)
    print(f"[04] wrote spectrum -> {out_path}")

    # Headline figures
    plot_spectrum(
        {"example": plot_data["example"]} if "example" in plot_data else {},
        save_to=figure_path("fig_active_subspace_spectrum_example"),
        title=f"Example-level active subspace spectrum  (layer {layer_index})",
    )
    plot_spectrum(
        {"span": plot_data["span"]} if "span" in plot_data else {},
        save_to=figure_path("fig_active_subspace_spectrum_span"),
        title=f"Span-level active subspace spectrum  (layer {layer_index})",
    )
    high_low = {}
    if "high_v4" in plot_data: high_low["high_v4"] = plot_data["high_v4"]
    if "low_v4"  in plot_data: high_low["low_v4"]  = plot_data["low_v4"]
    if high_low:
        plot_spectrum(
            high_low,
            save_to=figure_path("fig_high_vs_low_sensitivity_spectrum"),
            title=f"High vs low v4-sensitivity spans  (layer {layer_index})",
        )

    # Console summary
    for name, (S, expl, cum) in plot_data.items():
        ks_report = [4, 8, 16, 32, 64]
        vals = []
        for k in ks_report:
            if k <= len(cum):
                vals.append(f"k={k}:{cum[k-1]:.3f}")
            else:
                vals.append(f"k={k}:{cum[-1]:.3f}")
        print(f"  {name:>10s}  " + "  ".join(vals))


if __name__ == "__main__":
    main()
