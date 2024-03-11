# Pretrain模型训练，单节点8卡
# --gradient_checkpointing 是在训练深度学习模型时启用了一种内存优化技术，称之为梯度检查点
# 用于大型模型的训练，尤其是在GPU受限的情况下，可以节省大量显卡内存。

# 当模型太大了无法一次性将整个前向传播和反向传播过程中所需要的中间变量存储在GPU内存中的时候，
# 梯度检查点技术可以用来解决这个问题，它可以临时保存和丢弃部分计算图中的中间结果。
# 具体来说，它会在前向传播过程中选择性地保存某些层的输出，在反向传播时重新计算那些已经丢弃
# 的输出以恢复必要的梯度信息，而不是直接从缓存中读取。

# 启用此选项后，模型会在训练期间动态计算梯度，通过牺牲一定的计算效率换取对显存占用的显著减少，
# 从而允许在有限资源下增大批次大小或者训练更大规模的模型。

# 要注意ds_zero_stage参数，它用来指定ZeRO优化器的阶段，是一个高级特性，微软还是牛啊！
# 能够实现模型参数、梯度参数、优化器参数状态的分片，分布到不同的GPU上去，一下子就解决了
# 显存不足的问题，而且还能实现模型并行。
deepspeed --num_nodes=1 --num_gpus=8 dxm_llm_main.py \
    --train_mode pretrain \
    --model_name_or_path ./Llama-2-7b-hf \
    --save_name model/model-pretrained \
    --data_path data/FinCorpus_tokenized \
    --epochs 1 \
    --per_device_train_batch_size 4 \
    --max_length 4096 \
    --ds_zero_stage 2 \
    --log_steps 2 \
    --save_steps 40 \
    --gradient_checkpointing
