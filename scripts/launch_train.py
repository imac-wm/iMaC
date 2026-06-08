import argparse
import os
import shutil
import tempfile

import paths
import torch


def _bootstrap_tmpdir_for_launch():
    candidates = [
        os.environ.get("TMPDIR"),
        "/dev/shm/torch_mp",
        "/tmp/torch_mp",
    ]
    selected = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            os.makedirs(candidate, exist_ok=True)
            if shutil.disk_usage(candidate).free < 256 * 1024 * 1024:
                continue
            selected = candidate
            break
        except Exception:
            continue

    if selected is not None:
        os.environ["TMPDIR"] = selected
        os.environ["TMP"] = selected
        os.environ["TEMP"] = selected
        tempfile.tempdir = selected


def _bootstrap_mp_sharing_strategy():
    strategy = os.environ.get("TORCH_MP_SHARING_STRATEGY", "file_descriptor")
    if not strategy:
        return
    try:
        current = torch.multiprocessing.get_sharing_strategy()
        if current != strategy:
            torch.multiprocessing.set_sharing_strategy(strategy)
    except Exception as exc:
        print(f"[launch_train] warning: failed to set torch mp sharing strategy={strategy}: {exc}")


PRESET_CONFIGS = {
    "baseline_rnd_mix_stage_one_alltask": "iMac.configs.baseline_wm_rnd_mix_stage_one_alltask.config",
    "baseline_rnd_mix_stage_two_alltask": "iMac.configs.baseline_wm_rnd_mix_stage_two_alltask.config",
    "worldarena_3d": "iMac.configs.0501_worldarena_3d_r1c120.config",
}


def main():
    parser = argparse.ArgumentParser(description="Launch WM training with explicit config or built-in presets.")
    parser.add_argument("--config_path", type=str, default=None, help="Full python config path, e.g. pkg.file.config")
    parser.add_argument("--preset", choices=PRESET_CONFIGS.keys(), default=None, help="Shortcut preset for common training runs")
    args = parser.parse_args()

    if args.config_path is None and args.preset is None:
        raise ValueError("Either --config_path or --preset must be provided.")

    _bootstrap_tmpdir_for_launch()
    _bootstrap_mp_sharing_strategy()

    config_path = args.config_path or PRESET_CONFIGS[args.preset]
    from giga_train import launch_from_config

    print(f"[launch_train] using config: {config_path}")
    print(f"[launch_train] TMPDIR={os.environ.get('TMPDIR')}")
    print(f"[launch_train] torch mp sharing strategy={torch.multiprocessing.get_sharing_strategy()}")
    launch_from_config(config_path)


if __name__ == "__main__":
    main()
