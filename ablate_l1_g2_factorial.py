r"""YOLO11-RGBD: exhaustive L1 output-channel interaction experiment (zero training).

Evaluate every combination of selected channels at {0.5, 1.0}, in TWO
separately controlled backgrounds:

* half_***: nonselected L1 output channels remain at 0.5; baseline half_000
  is the previously validated L1 all-channel *0.5 configuration.
* full_***: nonselected L1 output channels remain at 1.0; baseline full_111
  is the unmodified original checkpoint.

For --channels 9 13 15, the three digits (in that order) mean:
    0 = channel output multiplied by base-keep (default 0.5)
    1 = channel output multiplied by restore-keep (default 1.0)
E.g., half_010 restores ONLY channel 13 to 1.0; all other L1 channels
remain at 0.5.  full_011 attenuates ONLY channel 9 to 0.5.

All cases are independent fresh checkpoint runs. Scales affect post-L1
activations, NOT Conv weights or training. NME uses same-frame matching.

Required beside script: ablate_c2psa.py, ablate_layers.py,
  ablate_l1_channel_groups.py, ablate_l1_g2_detail.py.

Windows CMD:
  .venv\Scripts\python.exe ablate_l1_g2_factorial.py --channels 9 13 15 --mode both
"""
from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ablate_c2psa as ac
from ablate_layers import load_items
from ablate_l1_channel_groups import (
    check_l1_and_get_channels, changes_against, fmt, hits_map
)
from ablate_l1_g2_detail import Case, L1ChannelSelectScaler, make_paired_rows

LAYER_INDEX = 1
K = ac.KEYPOINT_COUNT


@dataclass(frozen=True)
class Plan:
    name: str
    context: str
    bits: tuple[int, int, int]
    scales: tuple[float, ...]
    reference: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Exhaustive 3-channel factorial L1 attenuation (no training)')
    p.add_argument('--dataset-root', default=str(ac._DEFAULT_DATASET))
    p.add_argument('--weights-dir', default=str(ac._REPO / 'train_weight'))
    p.add_argument('--weights-name', default='yolo11n_rgbd_4ch')
    p.add_argument('--split', default='test')
    p.add_argument('--conf', type=float, default=0.05)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--match-iou', type=float, default=0.50)
    p.add_argument('--expected-count', type=int, default=100)
    p.add_argument('--channels', nargs=3, type=int, default=[9, 13, 15],
                   metavar=('CH_A', 'CH_B', 'CH_C'),
                   help='Three zero-based L1 output-channel indices; default 9 13 15')
    p.add_argument('--base-keep', type=float, default=0.50)
    p.add_argument('--restore-keep', type=float, default=1.00)
    p.add_argument('--mode', choices=['both', 'half', 'full'], default='both')
    p.add_argument('--output-dir', default=str(ac._REPO / 'runs' / 'ablate_l1_g2_factorial'))
    return p.parse_args()


def build_plans(channels: int, selected: tuple[int, int, int], half: float,
                full: float, mode: str) -> list[Plan]:
    if len(selected) != 3 or len(set(selected)) != 3:
        raise ValueError('Exactly 3 distinct zero-based channels are required')
    if any(ch < 0 or ch >= channels for ch in selected):
        raise ValueError(f'Selected output channels {selected} are outside [0, {channels})')
    if not (0 <= half < full <= 1):
        raise ValueError('Require 0 <= base-keep < restore-keep <= 1')
    result = []
    for context in ['half', 'full']:
        if mode not in ('both', context):
            continue
        background = half if context == 'half' else full
        reference = 'half_000' if context == 'half' else 'full_111'
        for bits in itertools.product((0, 1), repeat=3):
            scales = [background] * channels
            for idx, bit in enumerate(bits):
                scales[selected[idx]] = full if bit else half
            name = context + '_' + ''.join(str(b) for b in bits)
            result.append(Plan(name, context, tuple(bits), tuple(scales), reference))
    return result


def apply_plan(model: Any, plan: Plan, channel_count: int, half: float, full: float) -> None:
    """Attach a checked L1 output scaler without changing trained weights."""
    current = check_l1_and_get_channels(model)
    if current != channel_count:
        raise RuntimeError(f'L1 output channels changed: {current} != {channel_count}')
    if all(s == full for s in plan.scales):
        # Truly unmodified original forward for full_111.
        return
    source = model.model.model[LAYER_INDEX]
    # Reuse the tested wrapper; use half as a universal base and override
    # EACH output channel to its planned absolute scale. The wrapper accepts
    # all-channel changes, ensuring exact per-channel factors in both contexts.
    spec = Case(plan.name, plan.context, tuple(range(channel_count)), half, full)
    scaler = L1ChannelSelectScaler(source, channel_count, spec)
    import torch
    with torch.no_grad():
        scaler.scales.copy_(torch.tensor(plan.scales, dtype=scaler.scales.dtype).reshape(1, channel_count, 1, 1))
    model.model.model[LAYER_INDEX] = scaler


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as fp:
        out = csv.DictWriter(fp, fieldnames=fields, extrasaction='ignore')
        out.writeheader()
        out.writerows(rows)


def mean(items: list[float]) -> float | None:
    return ac.mean_of(items)


def factorial_effects(context: str, available: dict[str, dict[str, dict[str, Any]]],
                      selected: tuple[int, int, int], common: list[str]) -> list[dict[str, Any]]:
    """Main / pairwise interaction effects using SAME images in all 8 cases.

    All contrasts are on mean NME / per-keypoint error, rather than model
    weights or layer outputs. A positive contrast means higher error.
    """
    if len(common) == 0:
        return []
    names = {bits: context + '_' + ''.join(map(str, bits))
             for bits in itertools.product((0, 1), repeat=3)}
    metrics = ['nme'] + [f'k{k}_px' for k in range(K)]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        average = {bits: mean([available[names[bits]][stem][metric] for stem in common])
                   for bits in names}
        for axis, ch in enumerate(selected):
            pos = mean([v for bits, v in average.items() if bits[axis] == 1])
            neg = mean([v for bits, v in average.items() if bits[axis] == 0])
            rows.append({'context': context, 'metric': metric,
                         'effect_type': 'main_1_minus_0', 'channels': str(ch),
                         'common_frames': len(common),
                         'effect': (pos - neg if pos is not None and neg is not None else None)})
        for a, b in itertools.combinations(range(3), 2):
            other = next(x for x in range(3) if x not in (a, b))
            interactions = []
            for o in (0, 1):
                vals = {}
                for bits in names:
                    if bits[other] == o:
                        vals[(bits[a], bits[b])] = average[bits]
                if all(vals[v] is not None for v in [(1, 1), (1, 0), (0, 1), (0, 0)]):
                    interactions.append(vals[(1, 1)] - vals[(1, 0)] - vals[(0, 1)] + vals[(0, 0)])
            rows.append({'context': context, 'metric': metric,
                         'effect_type': 'pairwise_difference_in_differences',
                         'channels': f'{selected[a]},{selected[b]}',
                         'common_frames': len(common), 'effect': mean(interactions)})
    return rows


def main() -> None:
    args = parse_args()
    selected = tuple(args.channels)
    root = Path(args.dataset_root).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    weight = Path(args.weights_dir).expanduser() / (args.weights_name.removesuffix('.pt') + '.pt')
    if not weight.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {weight}')
    if ac._LOCKED_VERSION != '8.4.175':
        raise RuntimeError(f'Expected Ultralytics 8.4.175, got {ac._LOCKED_VERSION}')

    rgb = ac.stem_map(root / 'rgb' / 'images' / args.split)
    rgbd = ac.stem_map(root / 'rgbd' / 'images' / args.split)
    labeldir = root / 'rgb' / 'labels' / args.split
    if not labeldir.is_dir():
        labeldir = root / 'rgbd' / 'labels' / args.split
    if not labeldir.is_dir():
        raise FileNotFoundError(f'GT directory not found: {labeldir}')
    stems = sorted(s for s in (rgb.keys() & rgbd.keys()) if (labeldir / f'{s}.txt').is_file())
    if not stems or (args.expected_count and len(stems) != args.expected_count):
        raise RuntimeError(f'Expected {args.expected_count} common images, got {len(stems)}')
    items = load_items(root, args.split, 4, stems)
    probe = ac.YOLO(str(weight))
    channel_count = check_l1_and_get_channels(probe)
    del probe
    gc.collect()
    plans = build_plans(channel_count, selected, args.base_keep, args.restore_keep, args.mode)
    contexts = sorted(set(p.context for p in plans))

    print('=' * 113)
    print('L1 OUTPUT CHANNEL FACTORIAL | 3 selected channels, 2 levels, independently controlled backgrounds')
    print(f'weights={weight} | frames={len(stems)} | ultralytics={ac._LOCKED_VERSION}')
    print(f'conf={args.conf} imgsz={args.imgsz} match_iou={args.match_iou} | L1 channels={channel_count}')
    print(f'selected (zero-based)={selected}; bit 0 -> {args.base_keep}, bit 1 -> {args.restore_keep}')
    print(f'other channels: half background -> {args.base_keep}; full background -> {args.restore_keep}')
    print(f'cases={len(plans)}; separate fresh checkpoint per run; no training')
    print('=' * 113)

    per_case: dict[str, list[dict[str, Any]]] = {}
    summary: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    for plan in plans:
        model = ac.YOLO(str(weight))
        apply_plan(model, plan, channel_count, args.base_keep, args.restore_keep)
        start = time.perf_counter()
        try:
            rows = ac.run_condition(model, items, args.conf, args.imgsz, args.match_iou)
        finally:
            del model
            gc.collect()
        per_case[plan.name] = rows
        hits = len(hits_map(rows))
        row: dict[str, Any] = {
            'case': plan.name, 'context': plan.context,
            'reference': plan.reference,
            'ch_a': selected[0], 'ch_b': selected[1], 'ch_c': selected[2],
            'bits': ''.join(map(str, plan.bits)),
            'other_keep': args.base_keep if plan.context == 'half' else args.restore_keep,
            'n_frames': len(stems), 'hits': hits,
            'nme_matched_only': mean([r['nme'] for r in rows]),
            'mean_boxes': mean([r['n_boxes'] for r in rows]),
            'elapsed_s': round(time.perf_counter() - start, 2),
        }
        for k in range(K):
            row[f'k{k}_px_matched_only'] = mean([r[f'k{k}_px'] for r in rows])
        summary.append(row)
        per_image.extend([{'case': plan.name, 'context': plan.context, **r} for r in rows])
        print(f"{plan.name:10} hits={hits:3}/{len(stems)} NME={fmt(row['nme_matched_only'])} {row['elapsed_s']:.1f}s")

    paired: list[dict[str, Any]] = []
    for row in summary:
        name = row['case']
        reference = row['reference']
        metrics = changes_against(per_case[reference], per_case[name], 'paired')
        row.update(metrics)
        if name != reference:
            paired.extend(make_paired_rows(name, reference, per_case[reference], per_case[name]))
    print('\nSAME-FRAME comparison against each context baseline')
    print(f"{'case':10} {'hits':>5} {'paired':>6} {'ref_nme':>10} {'case_nme':>10} {'delta%':>9} {'wins':>5} {'dK4':>9} {'dK5':>9} {'dK6':>9}")
    for row in summary:
        print(f"{row['case']:10} {row['hits']:5} {row['paired_paired']:6} "
              f"{fmt(row['paired_ref_nme']):>10} {fmt(row['paired_case_nme']):>10} "
              f"{fmt(row['paired_delta_pct'],2):>9} {row['paired_win_images']:5} "
              f"{fmt(row['paired_k4_delta_px'],2):>9} "
              f"{fmt(row['paired_k5_delta_px'],2):>9} "
              f"{fmt(row['paired_k6_delta_px'],2):>9}")

    matched_maps = {name: hits_map(rows) for name, rows in per_case.items()}
    matched_all = sorted(set.intersection(*(set(d) for d in matched_maps.values())))
    same_summary: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    for context in contexts:
        names = [p.name for p in plans if p.context == context]
        common = sorted(set.intersection(*(set(matched_maps[name]) for name in names)))
        print(f'\n{context} context: matched by all {len(names)} conditions = {len(common)}/{len(stems)}')
        factor_rows.extend(factorial_effects(context, matched_maps, selected, common))
        for plan in [p for p in plans if p.context == context]:
            name = plan.name
            for common_scope, commons in [('context', common), ('all', matched_all)]:
                entry = {
                    'case': name, 'context': context, 'common_scope': common_scope,
                    'common_frames': len(commons),
                    'nme_common': mean([matched_maps[name][s]['nme'] for s in commons]),
                }
                for k in range(K):
                    entry[f'k{k}_px_common'] = mean([matched_maps[name][s][f'k{k}_px'] for s in commons])
                same_summary.append(entry)
    print(f'All {len(plans)} conditions matched intersection: {len(matched_all)}/{len(stems)}')

    # Sanity check against previously measured controls; warn, don't hide results.
    def warn_baseline(name: str, expected: float):
        record = next((r for r in summary if r['case'] == name), None)
        if record and (record['hits'] != 100 or abs(record['nme_matched_only'] - expected) > 0.0005):
            print(f'WARNING: {name} differs from historical reference (100/100 NME {expected:.6f}).')
    warn_baseline('half_000', 0.059823)
    warn_baseline('full_111', 0.070292)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = ['case', 'context', 'reference', 'ch_a', 'ch_b', 'ch_c', 'bits',
                      'other_keep', 'n_frames', 'hits', 'nme_matched_only', 'mean_boxes', 'elapsed_s']
    summary_fields += [f'k{k}_px_matched_only' for k in range(K)]
    summary_fields += ['paired_paired', 'paired_ref_nme', 'paired_case_nme', 'paired_delta_nme',
                       'paired_delta_pct', 'paired_win_images', 'paired_lose_images']
    for k in range(K):
        summary_fields += [f'paired_k{k}_ref_px', f'paired_k{k}_case_px',
                           f'paired_k{k}_delta_px', f'paired_k{k}_win_images']
    summary_fields += ['paired_k456_delta_px', 'paired_k3456_delta_px']
    write_csv(out_dir / 'summary.csv', summary, summary_fields)
    write_csv(out_dir / 'per_image.csv', per_image,
              ['case', 'context', 'image', 'n_boxes', 'matched', 'iou', 'nme'] + [f'k{k}_px' for k in range(K)])
    paired_fields = ['case', 'reference', 'image', 'paired', 'reference_matched', 'case_matched',
                     'reference_nme', 'case_nme', 'delta_nme', 'case_iou', 'case_boxes']
    for k in range(K):
        paired_fields += [f'reference_k{k}_px', f'case_k{k}_px', f'delta_k{k}_px']
    write_csv(out_dir / 'paired_per_image.csv', paired, paired_fields)
    write_csv(out_dir / 'same_frames_summary.csv', same_summary,
              ['case', 'context', 'common_scope', 'common_frames', 'nme_common'] +
              [f'k{k}_px_common' for k in range(K)])
    write_csv(out_dir / 'factorial_effects.csv', factor_rows,
              ['context', 'metric', 'effect_type', 'channels', 'common_frames', 'effect'])
    config = {
        'dataset_root': str(root), 'split': args.split, 'weights': str(weight),
        'frames': len(stems), 'layer_index': LAYER_INDEX, 'layer_output_channels': channel_count,
        'selected_channels_0based': list(selected), 'bit_levels': {'0': args.base_keep, '1': args.restore_keep},
        'half_background_other_channels': args.base_keep,
        'full_background_other_channels': args.restore_keep,
        'condition_names': [p.name for p in plans], 'mode': args.mode,
        'conf': args.conf, 'imgsz': args.imgsz, 'match_iou': args.match_iou,
        'ultralytics': ac._LOCKED_VERSION,
        'evaluator': ac._EVAL_SCRIPT.name,
        'same_frames_in_each_context': {
            context: len(set.intersection(*(set(matched_maps[p.name]) for p in plans if p.context == context)))
            for context in contexts
        },
        'all_condition_common_frames': len(matched_all),
        'note': 'Output-channel scaling after L1 forward; each case starts from fresh .pt. '
                'Main and interaction effects are descriptive NME contrasts on common frames only.'
    }
    (out_dir / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\nSaved to: {out_dir}')
    print('summary.csv | per_image.csv | paired_per_image.csv | same_frames_summary.csv | factorial_effects.csv | run_config.json')


if __name__ == '__main__':
    main()
