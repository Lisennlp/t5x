
bash install.sh


YOUR_BUDGET_NAME='common_datasets'
TFDS_DATA_DIR='gs://common_datasets'
MODEL_DIR='./'
export GOOGLE_CLOUD_BUCKET_NAME=${YOUR_BUDGET_NAME} \
export TFDS_DATA_DIR=gs://${YOUR_BUDGET_NAME} \
export MODEL_DIR=gs://${YOUR_BUDGET_NAME}/OpenMoe0202/training \

python3  t5x/t5x/train.py \
        --gin_file="t5x/t5x/examples/t5/t5_1_1/examples/openmoe_base.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}


python3.10

改默认词表
/home/lishengping/miniconda3/envs/moe/lib/python3.10/site-packages/t5-0.9.4-py3.10.egg/t5/data/utils.py
or /home/lishengping/miniconda3/lib/python3.10/site-packages/t5-0.9.4-py3.10.egg/t5/data/utils.py
DEFAULT_SPM_PATH = "gs://common_datasets/vocab/c4_en_301_5Mexp_spm.model"
# 数据集注册文件
/home/lishengping/miniconda3/envs/py310moe/lib/python3.10/site-packages/t5-0.9.4-py3.10.egg/t5/data/tasks.py

c4数据集：c4_v020_unsupervised

配置文件
seqio.SentencePieceVocabulary.sentencepiece_model_file = "gs://common_datasets/vocab/c4_en_301_5Mexp_spm.model"
MIXTURE_OR_TASK_NAME = "c4_v020_unsupervised"



flaxformer/flaxformer/t5x/configs/moe/models/st_moe_decoder_only_base.gin
moe_architecture.SparseDecoderLayer.extra_mlp = None
Decoder
decode_from_continuous_inputs
TransparentLayerSequence

配置文件路径：/Users/lishengping/codes/jax_projects/t5x/t5x/examples/t5/t5_1_1/examples/openmoe_base.gin


# 1, 1, 8
self.mesh: Mesh(device_ids=array([[[0, 1, 4, 5, 2, 3, 6, 7]]]), axis_names=('data', 'expert', 'model'))
self._logical_axis_rules: (('batch', ('expert', 'data')), ('vocab', 'model'), ('mlp', 'model'), ('heads', 'model'), ('kv', None), ('joined_kv', 'model'), ('embed', 'model'), ('relpos_buckets', None), ('abspos_buckets', None), ('length', None), ('layers', None), ('stack', None), ('mlp_activations', None), ('expert', 'expert'), ('expert_mlp', 'model'), ('expert_replicas', 'data'), ('unmodeled', None))


# 1, 8, 1
mesh = ('data', 'expert', 'model')
self._logical_axis_rules = (('batch', ('expert', 'data')), ('vocab', 'model'), ('mlp', 'model'), ('heads', 'model'), ('kv', None), ('joi
ned_kv', 'model'), ('embed', 'model'), ('relpos_buckets', None), ('abspos_buckets', None), ('length', None), ('layers', None), ('stack'
, None), ('mlp_activations', None), ('expert', 'expert'), ('expert_mlp', 'model'), ('expert_replicas', 'data'), ('unmodeled', None))