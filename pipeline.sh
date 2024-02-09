#!/bin/bash 
#SBATCH --job-name=HB1_GB1_GB2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=30GB
#SBATCH --time=48:00:00
#SBATCH --output /home/echiavassa/deep-visual-geo-localization-benchmark/outputs/5_10/HB1_GB1_GB2out.txt
#SBATCH --error /home/echiavassa/deep-visual-geo-localization-benchmark/outputs/5_10/HB1_GB1_GB2err.txt
ml purge
ml Python
source /home/echiavassa/tesivenv/bin/activate

DATASET_SUMMARY=HB1_GB1_GB2
NUMBER_OF_TRAINING_DATASETS=1
declare -a TRAINING_DATASETS=(HyundaiDepartmentStore/B1)
declare -a VALIDATION_DATASETS=(GangnamStation/B1)
declare -a TEST_DATASETS=(GangnamStation/B2)

HOME_PATH=/home/echiavassa
DATASETS_PATH=/home/echiavassa/vg_datasets/datasets/${DATASET_SUMMARY}/images


for ((i=0; i<${NUMBER_OF_TRAINING_DATASETS}; i++));
do
    TRAIN=train${i}
    SOURCE_PATH=${HOME_PATH}/${TRAINING_DATASETS[$i]}/release/mapping
    TARGET_PATH=${DATASETS_PATH}/${TRAIN}/database/${TRAINING_DATASETS[$i]}/release
    mkdir -p ${TARGET_PATH}
    ln -s ${SOURCE_PATH} ${TARGET_PATH}
    SOURCE_PATH=${HOME_PATH}/${TRAINING_DATASETS[$i]}/release/validation
    TARGET_PATH=${DATASETS_PATH}/${TRAIN}/queries/${TRAINING_DATASETS[$i]}/release
    mkdir -p ${TARGET_PATH}
    ln -s ${SOURCE_PATH} ${TARGET_PATH}
done
mkdir -p ${DATASETS_PATH}/val/database
mkdir -p ${DATASETS_PATH}/val/queries

for (( i=0; i<1; i++ ));
do
  SOURCE_PATH=${HOME_PATH}/${VALIDATION_DATASETS[$i]}/release/mapping
  TARGET_PATH=${DATASETS_PATH}/val/database/${VALIDATION_DATASETS[$i]}/release
  mkdir -p ${TARGET_PATH}
  ln -s ${SOURCE_PATH} ${TARGET_PATH}
  SOURCE_PATH=${HOME_PATH}/${VALIDATION_DATASETS[$i]}/release/validation
  TARGET_PATH=${DATASETS_PATH}/val/queries/${VALIDATION_DATASETS[$i]}/release
  mkdir -p ${TARGET_PATH}
  ln -s ${SOURCE_PATH} ${TARGET_PATH}
done

mkdir -p ${DATASETS_PATH}/test/database
mkdir -p ${DATASETS_PATH}/test/queries

for (( i=0; i<1; i++ ));
do
  SOURCE_PATH=${HOME_PATH}/${TEST_DATASETS[$i]}/release/mapping
  TARGET_PATH=${DATASETS_PATH}/test/database/${TEST_DATASETS[$i]}/release
  mkdir -p ${TARGET_PATH}
  ln -s ${SOURCE_PATH} ${TARGET_PATH}
  SOURCE_PATH=${HOME_PATH}/${TEST_DATASETS[$i]}/release/validation
  TARGET_PATH=${DATASETS_PATH}/test/queries/${TEST_DATASETS[$i]}/release
  mkdir -p ${TARGET_PATH}
  ln -s ${SOURCE_PATH} ${TARGET_PATH}
done

TRAIN_THRESHOLD=5
TEST_THRESHOLD=10
EXP_NAME=${TRAIN_THRESHOLD}_${TEST_THRESHOLD}/${DATASET_SUMMARY}

python train.py --datasets_folder /home/echiavassa/vg_datasets/datasets --dataset_name ${DATASET_SUMMARY} --save_dir ${EXP_NAME} \
        --queries_per_epoch 500 --cache_refresh_rate 500 --number_of_training_datasets ${NUMBER_OF_TRAINING_DATASETS} \
        --train_positives_dist_threshold ${TRAIN_THRESHOLD} --val_positive_dist_threshold ${TEST_THRESHOLD} \
        --backbone resnet18conv5 --aggregation gem --fc_output_dim 512 --num_workers 3
# --backbone, --aggregation and --fc_output_dim are not really needed as model is loaded from torch hub