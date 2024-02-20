import os

CONTENT = \
"""
source /home/echiavassa/tesivenv/bin/activate

DATASET_SUMMARY=DATASETS_STRING_SUMMARY
NUMBER_OF_TRAINING_DATASETS=DATASETS_STRING_LEN
declare -a TRAINING_DATASETS=(DATASET_STRING)
declare -a VALIDATION_DATASETS=(VAL_DATASET)
declare -a TEST_DATASETS=(TEST_DATASET_NAME)

HOME_PATH=/data/echiavassa
DATASETS_PATH=/home/echiavassa/vg_datasets/datasets/${DATASET_SUMMARY}_AGG/images


for ((i=0; i<${NUMBER_OF_TRAINING_DATASETS}; i++));
do
    TRAIN=train0
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

python train.py --datasets_folder /home/echiavassa/vg_datasets/datasets --dataset_name ${DATASET_SUMMARY} --save_dir EXP_NAME \
        --queries_per_epoch QUERIES_PER_EPOCH --cache_refresh_rate QUERIES_PER_EPOCH --number_of_training_datasets ${NUMBER_OF_TRAINING_DATASETS} \
        --train_positives_dist_threshold TRAIN_THRESHOLD --val_positive_dist_threshold VAL_THRESHOLD \
        --backbone resnet50conv5 --aggregation gem --fc_output_dim 2048 --num_workers 3 --train_batch_size 4 --load_from_hub \
        --infer_batch_size 1 --visualize_triplets
"""

dataset_short_names = {
    "HB1": "HyundaiDepartmentStore/B1",
    "H1F": "HyundaiDepartmentStore/1F",
    "H4F": "HyundaiDepartmentStore/4F",
    "GB1": "GangnamStation/B1",
    "GB2": "GangnamStation/B2",
    "BAI": "BaiduMall"
    }
dataset_queries = {
    "HB1": 1000,
    "H1F": 500,
    "H4F": 500,
    "GB1": 1000,
    "GB2": 500,
    "BAI": 500,
}
folder = "/home/echiavassa/deep-visual-geo-localization-benchmark"
model_name = "eigenplaces_resnet50_2048"
dataset_string_summaries = ["HB1-GB1_H4F_GB2"]
for dataset_string_summary in dataset_string_summaries:
    training_set_list = sorted(dataset_string_summary.split("_")[0].split("-"))
    dataset_string_len = str(len(training_set_list))
    queries_per_epoch = str(min(dataset_queries[key] for key in training_set_list))
    dataset_string = " ".join([dataset_short_names[key] for key in training_set_list])
    val_dataset = dataset_short_names[dataset_string_summary.split("_")[1]]
    test_dataset = dataset_short_names[dataset_string_summary.split("_")[2]]
    thresholds = ["10_25", "5_10", "2_5"]
    for threshold in thresholds[-1:]:
        train_threshold, val_threshold = threshold.split("_")
        exp_name = threshold + "/" + dataset_string_summary + "/" + model_name
        content = CONTENT.replace("DATASETS_STRING_SUMMARY", dataset_string_summary)\
                            .replace("DATASETS_STRING_LEN", dataset_string_len)\
                            .replace("DATASET_STRING", dataset_string)\
                            .replace("VAL_DATASET", val_dataset)\
                            .replace("TEST_DATASET_NAME", test_dataset)\
                            .replace("TRAIN_THRESHOLD", train_threshold)\
                            .replace("VAL_THRESHOLD", val_threshold)\
                            .replace("QUERIES_PER_EPOCH", queries_per_epoch)\
                            .replace("EXP_NAME", exp_name)
        filename = f"{folder}/jobs/{exp_name}_test.sh"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        os.makedirs(f"{folder}/outputs/{exp_name}", exist_ok=True)
        with open(filename, "w") as file:
            _ = file.write(content)
        _ = os.system(f"bash {filename}")
        print(f"bash {filename}")