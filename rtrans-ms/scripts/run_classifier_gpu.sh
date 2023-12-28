echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_classifier_gpu.sh DEVICE_ID"
echo "DEVICE_ID is optional, default value is zero"
echo "for example: bash scripts/run_classifier_gpu.sh DEVICE_ID 1"
echo "assessment_method include: [MCC, Spearman_correlation ,Accuracy]"
echo "=============================================================================================================="

if [ -z $1 ]
then
    export CUDA_VISIBLE_DEVICES=3
else
    export CUDA_VISIBLE_DEVICES="$1"
fi


mkdir -p ms_log
assessment_method=Accuracy # F1
CUR_DIR=`pwd`
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
# python ${PROJECT_DIR}/../run_classifier.py  \
#     --config_path="../../task_classifier_config.yaml" \
#     --device_target="GPU" \
#     --do_train="true" \
#     --do_eval="true" \
#     --assessment_method="Accuracy" \
#     --epoch_num=20 \
#     --num_class=15 \
#     --train_data_shuffle="true" \
#     --eval_data_shuffle="false" \
#     --train_batch_size=32 \
#     --eval_batch_size=1 \
#     --save_finetune_checkpoint_path="/sdb/nlp21/Project/LongDocClass/models-r2.0/saved_model_08060030" \
#     --load_pretrain_checkpoint_path="/sdb/nlp21/Project/LongDocClass/models-r2.0/bert-base-uncased/bert_base_uncased.ckpt" \
#     --load_finetune_checkpoint_path="/sdb/nlp21/Project/LongDocClass/models-r2.0/bert-base-uncased/bert_base_uncased.ckpt" \
#     --train_data_file_path="../../models-r2.0/tnews_public/train.tf_record" \
#     --eval_data_file_path="../../models-r2.0/tnews_public/eval.tf_record" \
#     --schema_file_path="" 
    #> classifier_ep10_08031013.log 2>&1 &

# Note: 记得修改config中的pos的位置编码的长度与你的文本的最大长度保持一致
# hyper
# python ${PROJECT_DIR}/../run_classifier.py  \
#     --config_path="../../task_classifier_config.yaml" \
#     --device_target="GPU" \
#     --do_train="true" \
#     --do_eval="true" \
#     --assessment_method=$assessment_method \
#     --epoch_num=10 \
#     --num_class=2 \
#     --train_data_shuffle="true" \
#     --eval_data_shuffle="false" \
#     --train_batch_size=16 \
#     --eval_batch_size=1 \
#     --save_finetune_checkpoint_path="/sdb/nlp21/Project/LongDocClass/models-r2.0/saved_model_08060030" \
#     --load_pretrain_checkpoint_path="/sdb/nlp21/Project/LongDocClass/models-r2.0/bert-base-uncased/bert_base_uncased.ckpt" \
#     --load_finetune_checkpoint_path="/sdb/nlp21/Project/LongDocClass/models-r2.0/bert-base-uncased/bert_base_uncased.ckpt" \
#     --train_data_file_path="../../models-r2.0/dataset/hyper/data_ms/train.mindrecord" \
#     --eval_data_file_path="../../models-r2.0/dataset/hyper/data_ms/dev.mindrecord" \
#     --schema_file_path="" >> classifier_hyper_ep30_08161030.log 2>&1 &
    # 
    # ==============================================================
    # Precision 0.813953 
    # Recall 0.921053 
    # F1 0.864198 
    # ==============================================================
    
# 20news
python ${PROJECT_DIR}/../run_classifier.py  \
    --config_path="../../task_classifier_config.yaml" \
    --device_target="GPU" \
    --do_train="true" \
    --do_eval="true" \
    --assessment_method=$assessment_method \
    --epoch_num=10 \
    --num_class=20 \
    --train_data_shuffle="true" \
    --eval_data_shuffle="false" \
    --train_batch_size=16 \
    --eval_batch_size=1 \
    --save_finetune_checkpoint_path="/sdb/nlp21/Project/LongDocClass/models-r2.0/saved_model_09030030" \
    --load_pretrain_checkpoint_path="/sdb/nlp21/Project/LongDocClass/models-r2.0/bert-base-uncased/bert_base_uncased.ckpt" \
    --load_finetune_checkpoint_path="/sdb/nlp21/Project/LongDocClass/models-r2.0/bert-base-uncased/bert_base_uncased.ckpt" \
    --train_data_file_path="../../models-r2.0/dataset/20news/data_ms/train.mindrecord" \
    --eval_data_file_path="../../models-r2.0/dataset/20news/data_ms/predict.mindrecord" \
    --schema_file_path="" >> classifier_20news_ep10_08161030.log 2>&1 &