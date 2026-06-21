#!/usr/bin/env bash
# Generate a read-only SparseWorld environment/data diagnostic report.
# Usage: bash tools/diagnose_sparseworld_env.sh [output_report.md]

set -u
set -o pipefail

OUT="${1:-sparseworld_env_report.md}"
ROOT="$(pwd)"
RUN_TIMEOUT="${SPARSEWORLD_DIAG_TIMEOUT:-60s}"

# Write both stdout and stderr into the markdown report while keeping the script
# non-fatal: many checks are expected to fail on incomplete environments.
exec >"${OUT}" 2>&1

section() {
  printf '\n\n## %s\n\n' "$1"
}

run() {
  local desc="$1"
  shift
  printf '\n### %s\n\n' "$desc"
  printf '```bash\n%s\n```\n\n' "$*"
  printf '```text\n'
  if command -v timeout >/dev/null 2>&1; then
    timeout "$RUN_TIMEOUT" "$@"
  else
    "$@"
  fi
  local code=$?
  printf '\n[exit_code=%s]\n' "$code"
  printf '```\n'
  return 0
}

check_path() {
  local p="$1"
  if [ -e "$p" ]; then
    printf 'OK      %s\n' "$p"
  elif [ -L "$p" ]; then
    printf 'BROKEN_SYMLINK %s -> %s\n' "$p" "$(readlink "$p" 2>/dev/null || true)"
  else
    printf 'MISSING  %s\n' "$p"
  fi
}

cat <<HEADER
# SparseWorld Environment Diagnostic Report

Generated at: $(date -Is 2>/dev/null || date)

Working directory: \\`$ROOT\\`

This report is read-only. It does not start training, download data, or modify repository files.
HEADER

section "0. Basic location"
run "Date" date
run "Host" hostname
run "User" whoami
run "PWD" pwd
run "Git top-level" bash -lc 'git rev-parse --show-toplevel 2>/dev/null || echo NOT_A_GIT_REPO_OR_GIT_UNAVAILABLE'
run "Git branch" bash -lc 'git branch --show-current 2>/dev/null || true'
run "Git status" bash -lc 'git status --short 2>/dev/null || true'

section "1. Top-level structure"
run "Top-level files and symlinks" bash -lc "find . -maxdepth 1 -mindepth 1 -printf '%M %u %g %s %p -> %l\\n' 2>/dev/null | sort | sed -n '1,200p'"

section "2. Symlink and path status"
for p in data work_dirs work-dir ckpts admlp occworld; do
  printf '\n### %s\n\n```text\n' "$p"
  if [ -e "$p" ] || [ -L "$p" ]; then
    ls -ld "$p" || true
    printf '\nreadlink -f: '
    readlink -f "$p" || true
    printf '\nstat:\n'
    stat "$p" || true
  else
    printf 'MISSING: %s\n' "$p"
  fi
  printf '```\n'
done

section "3. Disk and mount status"
run "Disk usage for repo" df -h .
for p in data work_dirs work-dir /mnt/mnt7; do
  if [ -e "$p" ] || [ -L "$p" ]; then
    run "Disk usage for $p" df -h "$p"
  fi
done
run "Mount info for /mnt/mnt7" bash -lc 'if [ -e /mnt/mnt7 ]; then df -h /mnt/mnt7; mountpoint /mnt/mnt7 || true; else echo /mnt/mnt7 missing; fi' 

section "4. GPU / CUDA / driver"
run "nvidia-smi" bash -lc 'nvidia-smi || true'
run "nvcc version" bash -lc 'nvcc --version || true'

section "5. Python / Conda / Pip"
run "Python executable" bash -lc 'which python || true'
run "Python version" bash -lc 'python -V || true'
run "Pip version" bash -lc 'which pip || true; pip -V || true'
run "Conda env" bash -lc 'echo "CONDA_PREFIX=${CONDA_PREFIX:-}"; echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"; if command -v conda >/dev/null 2>&1; then timeout 20s conda info --envs 2>/dev/null || true; else echo conda not found; fi'

section "6. Core Python package versions"
run "Package import/version check" bash -lc 'timeout 60s python - <<"PY"
import sys, importlib
print("python executable:", sys.executable)
print("python version:", sys.version.replace("\n", " "))
mods = ["torch", "torchvision", "mmcv", "mmdet", "mmseg", "mmdet3d", "numpy", "cv2", "nuscenes", "pyquaternion"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"{m}: version={getattr(mod, '__version__', 'NO __version__')} file={getattr(mod, '__file__', 'NO __file__')}")
    except Exception as e:
        print(f"{m}: IMPORT_FAIL: {type(e).__name__}: {e}")
try:
    import torch
    print("torch cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)
    print("torch device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i} name={torch.cuda.get_device_name(i)}")
except Exception as e:
    print("torch cuda check failed:", repr(e))
PY
'

section "7. Entrypoint and main config"
run "tools/dist_train.sh" bash -lc 'sed -n "1,120p" tools/dist_train.sh 2>/dev/null || echo MISSING tools/dist_train.sh'
run "sparseworld-traj-finetune.py important lines" bash -lc 'nl -ba configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py 2>/dev/null | sed -n "40,310p" || echo MISSING main config'

section "8. Required file existence"
printf '```text\n'
required_paths=(
  data
  data/nuscenes
  data/nuscenes/samples
  data/nuscenes/sweeps
  data/nuscenes/maps
  data/nuscenes/v1.0-trainval
  data/nuscenes/gts
  data/nuscenes/bevdetv2-nuscenes_infos_train.pkl
  data/nuscenes/bevdetv2-nuscenes_infos_val.pkl
  occworld
  occworld/nuscenes_infos_train_temporal_v3_scene.pkl
  occworld/nuscenes_infos_val_temporal_v3_scene.pkl
  admlp
  admlp/fengze_nuscenes_infos_train.pkl
  admlp/fengze_nuscenes_infos_val.pkl
  admlp/stp3_val
  admlp/stp3_val/data_nuscene.pkl
  admlp/stp3_val/filter_token.pkl
  admlp/stp3_val/stp3_occupancy.pkl
  admlp/stp3_val/stp3_traj_gt.pkl
  ckpts
  ckpts/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth
)
for p in "${required_paths[@]}"; do
  check_path "$p"
done
printf '```\n'

section "9. Required file sizes"
printf '```text\n'
for p in "${required_paths[@]}"; do
  if [ -f "$p" ]; then
    du -h "$p"
  fi
done
printf '```\n'

section "10. Data tree snapshot"
if [ -e data ]; then
  run "data tree maxdepth 3" bash -lc "find data -maxdepth 3 -mindepth 1 -printf '%M %u %g %s %p -> %l\n' 2>/dev/null | sort | sed -n '1,240p'"
else
  printf '
### data tree maxdepth 3

```text
MISSING data
```
'
fi
if [ -e occworld ]; then
  run "occworld tree" bash -lc "find occworld -maxdepth 2 -mindepth 1 -printf '%M %u %g %s %p -> %l\n' 2>/dev/null | sort | sed -n '1,120p'"
else
  printf '
### occworld tree

```text
MISSING occworld
```
'
fi
if [ -e admlp ]; then
  run "admlp tree maxdepth 3" bash -lc "find admlp -maxdepth 3 -mindepth 1 -printf '%M %u %g %s %p -> %l\n' 2>/dev/null | sort | sed -n '1,200p'"
else
  printf '
### admlp tree maxdepth 3

```text
MISSING admlp
```
'
fi
if [ -e ckpts ]; then
  run "ckpts tree" bash -lc "find ckpts -maxdepth 1 -mindepth 1 -printf '%M %u %g %s %p -> %l\n' 2>/dev/null | sort | sed -n '1,120p'"
else
  printf '
### ckpts tree

```text
MISSING ckpts
```
'
fi
section "11. nuScenes directory counts"
printf '```text\n'
for d in data/nuscenes/samples data/nuscenes/sweeps data/nuscenes/maps data/nuscenes/v1.0-trainval data/nuscenes/gts data/depth_gt data/seg_gt_lidarseg; do
  echo "---- $d ----"
  if [ -e "$d" ]; then
    echo "immediate entries: $(find "$d" -maxdepth 1 -mindepth 1 2>/dev/null | wc -l)"
    find "$d" -maxdepth 1 -mindepth 1 2>/dev/null | sort | sed -n '1,20p'
  else
    echo "MISSING"
  fi
  echo
done
printf '```\n'

section "12. PKL quick read check"
run "Read important pkl files" bash -lc 'timeout 60s python - <<"PY"
import os, pickle
paths = [
    "data/nuscenes/bevdetv2-nuscenes_infos_train.pkl",
    "data/nuscenes/bevdetv2-nuscenes_infos_val.pkl",
    "occworld/nuscenes_infos_train_temporal_v3_scene.pkl",
    "occworld/nuscenes_infos_val_temporal_v3_scene.pkl",
    "admlp/fengze_nuscenes_infos_train.pkl",
    "admlp/fengze_nuscenes_infos_val.pkl",
    "admlp/stp3_val/data_nuscene.pkl",
    "admlp/stp3_val/filter_token.pkl",
    "admlp/stp3_val/stp3_occupancy.pkl",
    "admlp/stp3_val/stp3_traj_gt.pkl",
]
for p in paths:
    print("\n----", p, "----")
    if not os.path.exists(p):
        print("MISSING")
        continue
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        print("type:", type(obj))
        if isinstance(obj, dict):
            print("keys:", list(obj.keys())[:20])
            if "infos" in obj:
                print("len(infos):", len(obj["infos"]))
                if obj["infos"]:
                    first = obj["infos"][0]
                    print("first info type:", type(first))
                    if isinstance(first, dict):
                        print("first info keys:", list(first.keys())[:30])
        elif isinstance(obj, list):
            print("len(list):", len(obj))
            if obj:
                print("first type:", type(obj[0]))
        else:
            try:
                print("len:", len(obj))
            except Exception:
                pass
    except Exception as e:
        print("READ_FAIL:", type(e).__name__, e)
PY
'

section "13. Project import check"
run "Import project modules" bash -lc 'PYTHONPATH="$(pwd):${PYTHONPATH}" python - <<"PY"
import os, sys
print("cwd:", os.getcwd())
print("sys.path[0:5]:", sys.path[:5])
try:
    import mmdet3d
    print("mmdet3d imported:", getattr(mmdet3d, "__file__", None))
except Exception as e:
    print("mmdet3d import failed:", type(e).__name__, e)
try:
    from mmdet3d.datasets import NuScenesDatasetOccpancy4DTraj
    print("NuScenesDatasetOccpancy4DTraj import OK")
except Exception as e:
    print("NuScenesDatasetOccpancy4DTraj import failed:", type(e).__name__, e)
try:
    from mmdet3d.datasets.pipelines import LoadOccGTFromFile4DTraj
    print("LoadOccGTFromFile4DTraj import OK")
except Exception as e:
    print("LoadOccGTFromFile4DTraj import failed:", type(e).__name__, e)
PY'

section "14. Config parse check"
run "Parse SparseWorld training config" bash -lc 'PYTHONPATH="$(pwd):${PYTHONPATH}" python - <<"PY"
try:
    from mmcv import Config
    cfg = Config.fromfile("configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py")
    print("Config parse OK")
    print("model type:", cfg.model.get("type"))
    print("dataset type:", cfg.data.train.get("type"))
    print("train ann_file:", cfg.data.train.get("ann_file"))
    print("val ann_file:", cfg.data.val.get("ann_file"))
    print("test ann_file:", cfg.data.test.get("ann_file"))
    print("samples_per_gpu:", cfg.data.get("samples_per_gpu"))
    print("workers_per_gpu:", cfg.data.get("workers_per_gpu"))
    print("load_from:", cfg.get("load_from"))
    print("runner:", cfg.get("runner"))
except Exception as e:
    print("Config parse FAILED:", type(e).__name__, e)
PY'

section "15. Final note"
if [[ "$OUT" = /* ]]; then
  printf 'Report path: `%s`\n' "$OUT"
else
  printf 'Report path: `%s/%s`\n' "$ROOT" "$OUT"
fi
