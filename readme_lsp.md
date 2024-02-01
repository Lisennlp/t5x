
YOUR_BUDGET_NAME='common_datasets'
TFDS_DATA_DIR='gs://common_datasets'
MODEL_DIR='./'
export GOOGLE_CLOUD_BUCKET_NAME=${YOUR_BUDGET_NAME} \
export TFDS_DATA_DIR=gs://${YOUR_BUDGET_NAME} \
export MODEL_DIR=gs://${YOUR_BUDGET_NAME}/openmoe_test/training \

python3  t5x/t5x/train.py \
        --gin_file="t5x/t5x/examples/t5/t5_1_1/examples/openmoe_base.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}


python3.10

pip install -f https://storage.googleapis.com/jax-releases/libtpu_releases.html jax[tpu]==0.4.16
改默认词表
/home/lishengping/miniconda3/envs/py310moe/lib/python3.10/site-packages/t5-0.9.4-py3.10.egg/t5/data/utils.py
/home/lishengping/miniconda3/lib/python3.10/site-packages/t5-0.9.4-py3.10.egg/t5/data/utils.py

# c4
c4_prefix_lm_objective_encoder_decoder_architecture

# 数据集注册
/home/lishengping/miniconda3/envs/py310moe/lib/python3.10/site-packages/t5-0.9.4-py3.10.egg/t5/data/tasks.py

数据集：c4_v020_unsupervised

/home/lishengping/projects/./t5x/t5x/train.py  # save model部分有问题

配置文件
seqio.SentencePieceVocabulary.sentencepiece_model_file = "gs://common_datasets/vocab/c4_en_301_5Mexp_spm.model"
MIXTURE_OR_TASK_NAME = "c4_v020_unsupervised"



base:
layer_nums=12
head_nums=12
model_dim=768
mlp_dim=2048
expert=16
TRAIN_EXPERT_CAPACITY_FACTOR=1.25

flaxformer/flaxformer/t5x/configs/moe/models/st_moe_decoder_only_base.gin
moe_architecture.SparseDecoderLayer.extra_mlp = None


Decoder
decode_from_continuous_inputs

TransparentLayerSequence