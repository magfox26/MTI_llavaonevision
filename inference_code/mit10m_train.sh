export CUDA_VISIBLE_DEVICES=1,2,3

NPROC_PER_NODE=3 \
CUDA_VISIBLE_DEVICES=1,2,3 \
swift sft \
--model /mnt/data/users/liamding/data/models/llava-onevision-qwen2-7b-si-hf \
--model_type llava_onevision_hf \
--train_type lora \
--dataset  /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/de_train_500.json \
           /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/en_train_500.json \
           /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/es_train_500.json \
           /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/fr_train_500.json \
           /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/it_train_500.json \
           /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/ja_train_500.json \
           /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/pt_train_500.json \
           /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/zh_train_500.json \
--val_dataset /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/de_val_10.json \
              /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/en_val_10.json \
              /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/es_val_10.json \
              /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/fr_val_10.json \
              /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/it_val_10.json \
              /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/ja_val_10.json \
              /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/pt_val_10.json \
              /mnt/data/users/liamding/data/dataset/MIT-10M/train/mit10_sample1/zh_val_10.json \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--learning_rate 2e-5 \
--lora_rank 16 \
--lora_alpha 32 \
--target_modules all-linear \
--gradient_accumulation_steps 8 \
--eval_steps 500 \
--save_strategy epoch \
--lora_dropout 0.2 \
--warmup_ratio 0.1 \
--logging_steps 10 \
--max_length 4096 \
--deepspeed /mnt/data/users/liamding/data/liu_SFT/swift/swift/llm/ds_config/zero2.json \
--dataloader_num_workers 4 \
--output_dir /mnt/data/users/liamding/data/liu_SFT/outcome_mit10m_sample500_new1
