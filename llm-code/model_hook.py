import time
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    LlamaForCausalLM, LlamaTokenizer,
    BloomForCausalLM, BloomTokenizerFast,
)
from torch.utils.data import DataLoader, DistributedSampler

from dataset import get_pt_dataset, DataCollatorForPT, JsonDatasetSFT
from dxm_llm_main import log_dist
from config import get_deepspeed_config, parse_arguments


def get_tokenizer(args):
    '''
        加载tokenizer
    '''
    # 对于llama系列模型使用LlamaTokenizer类
    if 'llama' in args.model_name_or_path.lower():
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    # 对于bloom系列模型使用BloomTokenizerFast类
    elif 'bloom' in args.model_name_or_path.lower():
        tokenizer = BloomTokenizerFast.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, fast_tokenizer=True)

    # 将分词器的pad_token设置为eos_token，以便正确处理填充（padding）
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model_common(args):
    """
        获取并加载模型文件
    """
    log_dist('=================== Loading Model =====================')
    log_dist(f"loading model from {args.model_name_or_path}")
    tic = time.time()

    # 对于llama系列模型使用 LlamaForCausalLM 类
    if 'llama' in args.model_name_or_path.lower():
        model = LlamaForCausalLM.from_pretrained(args.model_name_or_path)
    # 对于bloom系列模型使用 BloomForCausalLM 类
    elif 'bloom' in args.model_name_or_path.lower():
        model = BloomForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    log_dist(f'model loaded. costtime={time.time()-tic:.2f}s')
    log_dist(f"model = {model}")

    return model


def get_dataloader_common(args):
    '''
        用于创建数据加载器（DataLoader）和数据集
    '''
    tokenizer = get_tokenizer(args)

    log_dist(f'==================== Loading dataset =================')
    tic = time.time()
    if args.train_mode == 'pretrain':
        # 对于已预处理过的语料数据，直接使用load_from_disk()函数加载即可
        train_dataset = get_pt_dataset(args)
        collator = DataCollatorForPT(pad_token_id=tokenizer.pad_token_id)
    elif args.train_mode == 'sft':
        train_dataset = JsonDatasetSFT(args.data_path, tokenizer, args.max_length)
        collator = None
    else:
        raise ValueError(f"train_mode {args.train_mode} is not supported")

    # 进行数据集的分布式随机采样，确保在多GPU训练时数据的随机性
    # 在深度学习训练中，使用分布式计算环境时，DistributedSampler的作用至关重要。
    # 该类主要负责将大规模数据集均匀且合理地分配到多个计算节点或者GPU上
    # 
    # 1. 它会根据当前分布式环境中的进程数量，当前进程的rank自动划分数据集，确保每个进程得到不同的、不重复的子数据集
    # 2. train_data是原始数据集，
    # 3. shuffle=True表示随机采样，在每次epoch开始的时候，DistributedSampler会对整个数据集进行全局打乱，注意是全局视野下的打乱
    #    从而增强模型训练的随机性，防止模型过拟合，促进模型收敛
    # 4. seed=args.seed表示随机数种子，只要每次传的seed一样，就可以确保每次运行程序时数据集的划分方式以及打乱的顺序都一样
    #    从而使得实验结果具有可复现性，不至于每次运行结果都不一样，所以seed参数很重要的，每次都要设置一样的seed值
    sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed)

    '''
        DataLoader 类是一个用于加载数据并以批量方式提供给模型训练或验证的核心组件，也就是说他会从dataset里面一次性搞多条数据
        但是在从底层的数据集中提取数据时，是按顺序呢，还是按什么策略呢，这就要传递一个sampler的参数，由它来决定如何从底层的dataset里面抽样，这个参数是可选的
        2. 如果不指定，就根据shuffle参数来决定是否在每次epoch开始时对数据集进行全局打乱，从而增强模型训练的随机性
        3. 如果指定，就按sampler的策略来，也就说如果提供了自定义的sampler对象，它就会替代DataLoader内部的默认采样策略
        3. 提供不同的sampler实现，可以控制数据集中每个epoch中的遍历顺序，比如RandomSampler、分区间隔抽样SequentialSampler，或者
            分布式环境下的DistributedSampler等
        4. 在多机多卡训练时，DistributedSampler能确保每个服务器上的训练进程仅能够访问分配给它的那部分数据，从而实现数据的并行加载和训练；
        5. 还可以自己提供sampler，如果有特别的数据抽样需求，可以自己搞，一般不需要

    '''
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,   # 指定每个GPU上的batch大小，也就是每次送几条样本进GPU
        num_workers=16,  # 指定16个核并行处理
        sampler=sampler,
        drop_last=True,  # 指定是否将最后一个不完整的batch丢弃掉，如果设置为True，则最后一个batch的样本个数可能小于batch_size

        # 这个collate_fn参数在训练的时候用的挺多，指定如何把单个样本数据组合成一个批量数据，在使用DataLoader进行批量加载的时候，
        # 在每个批次加载之前调用这个函数：
        # 1. 格式化批量数据：默认情况下，collate_fn 将同一批次中的所有样本按照它们在数据集中的顺序堆叠起来（例如，对于图像数据，它会将多个图像张量在第一维度上进行堆叠；对于标签，则可能直接将标签列表连接起来）
        # 2. 处理异构数据：在自定义数据集包含不同形状或类型的数据时（如多模态输入、变长序列等），用户需要提供自定义的 collate_fn 来正确地组织和对齐这些数据，以便模型可以接受统一格式的输入。
        # 3. 执行额外预处理：此外，collate_fn 可以用于执行任何在数据送入模型前所需的额外批处理操作，比如填充变量长度序列到同一长度、计算注意力掩码、归一化数据等。
        collate_fn=collator,  # 自定义合并函数，处理批次内样本的打包方式
    )

    log_dist(f"Dataset Loaded: {args.data_path} costtime={time.time()-tic:.2f}s")
    log_dist(f"   Num samples: {len(train_dataset)}")
    log_dist(f"   Num Tokens: {len(train_dataset) * args.max_length / 1e9:.2f}B")
    log_dist(f"   Total Steps: {len(train_dataloader)}")

    return {
        "sampler": sampler,
        "train_dataloader": train_dataloader,
        "tokenizer": tokenizer
    }


def get_ds_config(args):
    '''
        用于获取 DeepSpeed 的配置参数
    '''
    ds_config = get_deepspeed_config(args)  # 获取deepspeed的配置参数，在config.py中定义
    return ds_config


def parse_args():
    '''
        解析命令行参数
    '''
    args = parse_arguments()  # 解析命令行参数的函数，在config.py中定义

    log_dist('============== 参数 ====================')
    for k, v in vars(args).items():
        log_dist(f'  {k} = {v}')
    log_dist('=======================================')

    return args


def get_op_lr(args, origin_model, dataloader_dict):
    '''
        获取优化器和学习率
    '''
    return None


def before_train(args, model_engine, dataloader_dict):
    '''
        在训练开始前执行
    '''
    pass


def on_step_end(args, model_engine, dataloader_dict, step_num, epoch_num, outputs):
    '''
        在每个训练步骤结束时执行
    '''
    pass


def on_epoch_end(args, model_engine, dataloader_dict, epoch_num):
    '''
        在每个训练周期（epoch）结束时执行
    '''
    pass


def after_train(args, model_engine, dataloader_dict):
    '''
        在整个训练过程结束时执行
    '''
    pass
