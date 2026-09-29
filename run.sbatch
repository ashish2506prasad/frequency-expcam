#!/bin/bash
#SBATCH --job-name=expcam_local
#SBATCH --output=slurm_logs/expcam_local-%j.log
#SBATCH --error=slurm_logs/expcam_local-%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00


echo "=========================================="
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE   : $(hostname)"
echo "GPUS   : $CUDA_VISIBLE_DEVICES"
echo "Start  : $(date)"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vae

ROOT=/home/medal/21d180009
DATA_ROOT=/home/medal/21d180009/trainX1_subset
OUT_DIR=/home/medal/21d180009/freq_expcam_fft

cd "$ROOT"
mkdir -p slurm_logs "$OUT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[paths] data-root=$DATA_ROOT"
echo "[paths] out=$OUT_DIR"

python -m expcam eval-set --where local --data-root "$DATA_ROOT" --force --out "$OUT_DIR"
python -m expcam 0 --config expcam/configs/fft.yaml --fft-metric energy_conf \
  --where local --data-root "$DATA_ROOT" --out "$OUT_DIR" \
  --backbones resnet18 vit_b_16 > freq_expcam_section0_fft.out 2>&1

echo "[zip] $OUT_DIR -> ${OUT_DIR}.zip"
python -c "import shutil, sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[1])" "$OUT_DIR"

echo "Finished at $(date)"
