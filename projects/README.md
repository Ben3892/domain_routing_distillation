# 实验启动与配置

## 本地启动

在仓库根目录运行。生成配置只需要 Python 3，CPU smoke 额外需要 NumPy。

```bash
python3 projects/run_experiment.py --matrix > experiment_matrix.json
python3 projects/run_experiment.py --preset domain --seed 42

python3 -m pip install numpy
python3 projects/run_experiment.py --matrix --smoke > smoke_results.json
```

CPU smoke 用同一批合成 token 和同一个固定 teacher，训练一个小型 bigram student。输出包含实际 KD 权重、按域验证 NTP CE、Coverage@k 和 OTMass。验证集由单独采样的合成样本构成，指标用于检查目标函数和路由行为。论文里的模型、训练数据和 benchmark 需要 GPU 训练环境。

## 实验矩阵

实验组由配置选择。所有核心对照使用 General / Math / Code 的 33 / 40 / 27 参考有效 token 配比，保持训练数据、初始化、训练预算和评测配置一致。

| 配置 | General | Math | Code | 比较目的 |
| --- | --- | --- | --- | --- |
| `lm` | LM | LM | LM | NTP 基线 |
| `kd` | KD | KD | KD | 前向 KL 基线 |
| `domain` | KD | LM | LM | 论文中的领域路由 |
| `matched-mix` | 0.67 LM + 0.33 KD | 同左 | 同左 | 固定全局 KD 权重，检验领域分配的作用 |
| `matched-random` | 按 token 随机选择 LM/KD | 同左 | 同左 | 固定全局硬路由 KD token 数，检验域标签是否优于随机分配 |

`--matrix` 还生成两个单因素扫描。Top-`k` 扫描取 1 / 4 / 16 / 64 / 256、固定 τ=1。温度扫描取 1 / 0.5 / 0.25、固定 `k=256`。两个扫描共享 `kd, k=256, τ=1`。单次配置可用 `python3 projects/run_experiment.py --preset kd --top-k 16 --temperature 0.5`。

CPU 参考实现按论文公式只对 teacher 截断分布应用温度，student 温度固定为 1。现有 Megatron fused loss 会同时调整 teacher 与 student 温度，训练 helper 还包含 τ² 缩放。跨实现对比时需要记录这三处设置。

真实训练建议用固定的有效 token 口径汇总全局和分域的实际 KD 权重，并从 `domain` 的实测 KD token 比例设定两个匹配对照。随机硬路由应使用固定 seed，明确是在全局、每步还是分域配额，并保持实际全局 KD token 数一致。验证统一记录按域 NTP CE、few-shot 和固定采样预算下的 Pass@K。若要复核论文的 token 级 OTMass/Entropy 路由，建议先在校准集匹配 KD token 比例，再用同一规则在固定验证集比较。

## Megatron GPU 复现入口

| 文件 | 作用 |
| --- | --- |
| [`run.sh`](../run.sh) | 原集群入口。依赖集群环境变量、hostfile、数据准备脚本和 PCIe 绑定工具。 |
| [`run_mcore_deepseek.sh`](run_mcore_deepseek.sh) | 历史训练命令和 45 个位置参数。包括模型规格、并行切分、目标权重、预算和输出路径。 |
| [`training_deepseek.py`](training_deepseek.py) | Megatron 训练入口，构造模型、dataloader 和 loss。 |
| [`hyper_parameter_predict`](../hyper_parameter_predict/README.md) | 独立的验证 loss 回归与候选配置打分。 |

当前仓库的 GPU 入口需要补齐以下组件后才能运行：

1. **运行环境。** 锁定可获得的 Megatron-LM checkout 与配套的 PyTorch/CUDA/NCCL、Transformer Engine 和所选 MoE kernel。仓库里的 `Megatron-LM` 是 gitlink，未提供 `.gitmodules`，原 wrapper 还引用了另一个目录名。需要明确 Megatron 源码与 `megatron_patch` 包的实际导入路径。
2. **模型与损失。** 训练入口需要 `megatron_patch.model.lightning_model.LightningModel`，checkpoint 保存入口为空，helper 依赖缺失的 teacher client 和选择性 loss 函数。先验证单步前反向、保存与重载，再验证领域路由、Top-`k` 和温度的实际生效情况。
3. **数据与初始化。** 提供 student Megatron checkpoint、对应 tokenizer、Weighted Dataset JSON，以及同词表且与训练位置对齐的离线 teacher 缓存。当前 `PTTeacherDataset` 的布局固定保存 Top-256。每条记录依次为 `seq_len+1` 个 int64 token、`seq_len × 256` 个 float32 log-prob 和等量 int32 token ID。扫更小的 `k` 可从排序好的 Top-256 筛出前缀后重新归一化，同时保留磁盘记录宽度。配置的 `training` 项包含 `file_path`、`val_file_path`、`name`、`weight`、`consumed_data_count` 和 `ref_loss`，`quality_id` 与 `file_size` 可选。当前 teacher dataset builder 将训练路径也用于验证和测试，正式评测前须接入真实分割。数据文件名的解析还依赖 `env_run/` 路径片段。
4. **集群入口。** 原 wrapper 硬编码了路径，包含全机进程匹配终止与后台 checkpoint 上传。先按实验环境重建启动层，再接入上面生成的实验配置。验证启动时记录实际的 TP/PP/CP/EP、数据并行大小、每步有效 token 数和独立输出目录。

原 launcher 会按 `floor(TRAIN_TOKENS / (GLOBAL_BATCH_SIZE × SEQ_LEN))` 换算训练步数，warmup 同理。计算短跑预算时先选定需要的步数，再换算 token 预算。

| 关键位置参数 | 需要填入的值 |
| --- | --- |
| 2–9 | 模型规格（论文 student 对应的候选为 `V24B`）、micro/global batch、学习率、序列长度和精度。同步核对模型规格与 checkpoint。 |
| 10–13 | TP / PP / CP / EP。wrapper 中 `V24B` 的 PP=4 预设为 12 / 9 / 9 / 6 层，配套检查 GPU 数、专家切分和 batch 整除。 |
| 21–26 | 训练和验证数据配置、输入 checkpoint、训练与 warmup token 预算、独立输出目录。 |
| 31、34–35、42–43 | 离线 teacher 模式 `local`、全局 LM/KD 权重、`teacher` 数据集类型、温度。领域权重、随机分配和 Top-`k` 前缀需要训练代码支持。 |

GPU 接通后的首轮检查应覆盖纯 LM、纯 KD 和领域路由，建议同一批数据跑 1–2 个训练 step，确认 loss 有限、General/Math/Code 的 KD 权重符合预设、验证使用独立数据，并能重载产出的 checkpoint。
