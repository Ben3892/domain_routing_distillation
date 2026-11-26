# 实验启动与配置

## 本地启动

在仓库根目录运行。生成配置只需要 Python 3，CPU smoke 额外需要 NumPy。

```bash
python3 projects/run_experiment.py --paper-matrix > paper_experiment_matrix.json
python3 projects/run_experiment.py --matrix > experiment_matrix.json
python3 projects/run_experiment.py --preset domain --seed 42

python3 -m pip install numpy
python3 projects/run_experiment.py --paper-matrix --smoke > paper_smoke_results.json
```

CPU smoke 用同一批合成 token 和同一个固定 teacher，训练一个小型 bigram student。输出包含实际 KD 权重、按域验证 NTP CE、Coverage@k、OTMass 和 teacher 熵。验证集由单独采样的合成样本构成，指标用于检查目标函数和路由行为。论文里的模型、训练数据和 benchmark 需要 GPU 训练环境。

## 论文复现协议

实验对应[论文](../Let_the_Data_Decide.pdf) §4–7。各实验从同一个 M₃（24B 总参数、4.5B 激活参数）checkpoint 热启，teacher 为 DeepSeek V3.1 Base。续训预算 100B tokens，global batch size 为 5760，sequence length 为 4096，常数学习率为 `5e-5`。General / Math / Code 的续训配比为 33 / 40 / 27。论文构建 M₃ 时使用的是 50 / 30 / 20 的配比。

| 论文实验 | 配置或方法 | 结果位置 |
| --- | --- | --- |
| §5 LM / KD | `--preset lm` / `--preset kd`。KD 为 Top-256、τ=1。 | 表 4–5 |
| §6.3 Top-k | `--preset kd --top-k K`，K 取 1、4、16、64、256，固定 τ=1。 | 表 6–7、图 1 |
| §6.4 温度 | `--preset kd --temperature T`，T 取 0.25、0.5、1，固定 Top-256。τ→0 的硬伪标签用 Top-1 表示，教师最大 logit 无并列时等价。 | 表 8–9、图 2 |
| §7.1 Domain | `--preset domain`。General 用 Top-256 KD、τ=1，Math / Code 用 LM。各域分别 packing，每条 packed sequence 只含一个域。 | 表 10–11 |
| §7.2 分域诊断 | 从训练语料抽取 1B tokens，固定 Top-256、τ=1，按域统计 Coverage@k、CondOTMass 与 Entropy。 | 图 3 |
| §7.3 OTMass | `--preset otmass --routing-threshold 0.3678`，另一组为 0.9876。观测 token 在截断 teacher 分布中的质量低于阈值时走 KD，未被 top-k 覆盖时质量为 0。 | 表 12–13 |
| §7.3 Entropy | `--preset entropy --routing-threshold 0.1`，另一组为 0.5。截断 teacher 分布的原始熵高于阈值时走 KD，使用自然对数。 | 表 14–15 |

`--paper-matrix` 生成 13 组唯一训练配置。温度和 Top-k 扫描共享两端的重复设置。OTMass 和 Entropy 都使用 Top-256、τ=1。论文未写明等于阈值时的处理，CPU 参考实现将它归到 LM。

评测按 §4.3 执行：

- 常规评测使用表 3 的 13 个 benchmark，沿用每项指定的 shot 数和指标，每项跑 3 次取均值。
- Pass@K 使用 MATH-500、AIME 2024、AIME 2025 和 HumanEval，执行 zero-shot，报告 K=1、16、64、128。评测采样固定为 top-k=50、top-p=0.95、temperature=1.0。记录模型 checkpoint、题库与判分版本、采样参数和每题有效样本数。
- 续训使用内部语料和 M₃ checkpoint。启动前还需准备原始语料版本、teacher 缓存、tokenizer 与 benchmark harness。CPU smoke 的诊断仅覆盖合成样本。

## 额外对照与配置

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

真实训练建议用固定的有效 token 口径汇总全局和分域的实际 KD 权重，并从 `domain` 的实测 KD token 比例设定两个匹配对照。随机硬路由应使用固定 seed，明确是在全局、每步还是分域配额，并保持实际全局 KD token 数一致。验证统一记录按域 NTP CE、few-shot 和固定采样预算下的 Pass@K。如需额外比较等 KD 预算下的 token 级 OTMass 与 Entropy 路由，可先在校准集调整阈值，再用固定规则跑对照。论文复现使用上表的固定阈值。

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
