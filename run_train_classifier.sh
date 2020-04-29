#!/usr/bin/env bash

SOURCE_PATH="${HOME}/Workspace/causal_disentanglement_miniproject"
AT="@"

# Test the job before actually submitting 
#SBATCH_OR_CAT=cat
SBATCH_OR_CAT=sbatch

for ld in 2 3 4 6 10; do

RUNS_PATH="${SOURCE_PATH}/models/CausalClassifier_ld${ld}"
echo $RUNS_PATH
mkdir -p $RUNS_PATH

"${SBATCH_OR_CAT}" << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%J_slurm.out"
#SBATCH --error="${RUNS_PATH}/%J_slurm.err"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="poklukar${AT}kth.se"
#SBATCH --constrain="khazadum|rivendell|shire|gondor"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB

echo "Sourcing conda.sh"
source "${HOME}/anaconda3/etc/profile.d/conda.sh"
echo "Activating conda environment"
conda activate JigsawPuzzle_VAE
nvidia-smi

python train_classifier.py \
        --classifier_config="CausalClassifier_ld${ld}"" \
        --vae_config="VAEConv2D_v2_CausalDsprite_ber_shape2_scale5_ld${ld}" \
        --generate_data=1 \
        --train=1 \
        --num_workers=0 \
        --cuda=1 
HERE
done