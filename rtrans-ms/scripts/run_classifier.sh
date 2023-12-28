echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_classifier.sh DEVICE_ID"
echo "DEVICE_ID is optional, default value is zero"
echo "for example: bash scripts/run_classifier.sh DEVICE_ID 1"
echo "assessment_method include: [MCC, Spearman_correlation ,Accuracy]"
echo "=============================================================================================================="

if [ -z $1 ]
then
    export DEVICE_ID=0
else
    export DEVICE_ID=$1
fi


mkdir -p ms_log
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python ${PROJECT_DIR}/../run_classifier.py  \
    --config_path="../../task_classifier_config.yaml" \
    --device_target="Ascend" \
    --do_train="true" \
    --do_eval="false" \
    --assessment_method="Accuracy" \
    --device_id=$DEVICE_ID \
    --epoch_num=3 \
    --num_class=2 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --train_batch_size=32 \
    --eval_batch_size=1 \
    --save_finetune_checkpoint_path="" \
    --load_pretrain_checkpoint_path="" \
    --load_finetune_checkpoint_path="" \
    --train_data_file_path="" \
    --eval_data_file_path="" \
    --schema_file_path="" > classifier_log.txt 2>&1 &
