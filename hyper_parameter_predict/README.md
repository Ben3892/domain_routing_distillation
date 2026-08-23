# 蒸馏超参数 → Validation Loss 回归

这个目录把一次完整训练实验看作一条样本，使用 4 个领域各自的 3 个蒸馏超参数（共 12 个输入特征）预测 `validation_loss`：

| 领域 | 输入特征 |
|---|---|
| STEM | `stem_temperature`, `stem_topk`, `stem_lm_loss_weight` |
| General | `general_temperature`, `general_topk`, `general_lm_loss_weight` |
| Math | `math_temperature`, `math_topk`, `math_lm_loss_weight` |
| Code | `code_temperature`, `code_topk`, `code_lm_loss_weight` |

支持 Random Forest、LightGBM、XGBoost 和 MLP。训练程序使用嵌套交叉验证比较模型，以 out-of-fold MAE 选择最佳模型，并导出超参数—预测 loss 的 partial-dependence 曲线。这里得到的是**预测关联**，不能直接解释为因果关系。

## 1. 数据格式

真实实验 CSV 至少需要以下列：

```text
experiment_id,stem_temperature,stem_topk,stem_lm_loss_weight,general_temperature,general_topk,general_lm_loss_weight,math_temperature,math_topk,math_lm_loss_weight,code_temperature,code_topk,code_lm_loss_weight,validation_loss
```

- 一行代表一次完整训练 run。
- `lm_loss_weight` 定义为总目标中 LM loss 的系数，范围 `[0, 1]`；如果你的日志保存的是 KD loss weight，请先换算并在全体实验中保持定义一致。
- 同一套超参数的不同随机种子使用相同 `experiment_id`。程序会通过 GroupKFold 把它们放在同一折，防止数据泄漏。
- `seed`、`run_id` 等追踪字段可以保留，但默认不会作为特征。
- 不要把训练结束后才知道的字段（训练 loss、最终 benchmark 分数、early-stop step 等）作为输入，否则会发生标签泄漏。
- 如果有多个验证目标，例如 `math_validation_loss` 和 `code_validation_loss`，分别运行训练命令并用 `--target` 指定目标。这样通常比直接混成一个 loss 更容易解释。

如果你的数据是一行一个领域（long format），也可以使用 `domain,temperature,topk,lm_loss_weight,validation_loss` 结构，并传 `--features domain,temperature,topk,lm_loss_weight`。`domain` 会被 one-hot 编码；同一次配置/seed 的各领域记录仍应使用同一个 `experiment_id`。

[示例训练数据](data/example_experiments.synthetic.csv) 是 720 行的**合成数据**，只用于验证代码流程，不能用于论文结论或真实超参数决策。[示例候选数据](data/example_candidates.synthetic.csv) 同样是合成的。

## 2. 安装

建议在本目录创建独立环境：

```bash
cd /Users/ben/git_projects/distillation_research/hyper_parameter_predict
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[boosting,dev]'
```

如果暂时不装 LightGBM/XGBoost，可以执行 `python -m pip install -e '.[dev]'`，并在训练时传 `--models random_forest,mlp`。完整依赖也记录在 `requirements.txt`。

macOS 上 LightGBM 还可能需要 OpenMP 运行库；若报错找不到 `libomp.dylib`，先执行 `brew install libomp`。某个可选增强模型无法导入时，训练程序会记录原因并跳过它，其余模型仍会继续。

## 3. 训练和比较模型

先用合成数据跑通：

```bash
distill-hparam-train \
  --data data/example_experiments.synthetic.csv \
  --output-dir outputs/example \
  --models random_forest,lightgbm,xgboost,mlp \
  --cv-folds 5 \
  --search-iterations 16 \
  --n-jobs -1
```

换成真实数据时只需替换 `--data`。默认使用上述 12 个特征和 `validation_loss` 目标。加入训练 token 数、学生规模、教师版本等实验开始前已知的变量时，显式传入完整列表：

```bash
distill-hparam-train \
  --data data/my_real_experiments.csv \
  --target validation_loss \
  --features stem_temperature,stem_topk,stem_lm_loss_weight,general_temperature,general_topk,general_lm_loss_weight,math_temperature,math_topk,math_lm_loss_weight,code_temperature,code_topk,code_lm_loss_weight,student_model,training_tokens \
  --group-column experiment_id \
  --output-dir outputs/real_v1
```

数值特征会做中位数填补和标准化；字符串特征会做众数填补和 one-hot 编码。未见过的新类别在预测时可安全处理。

主要输出如下：

| 文件 | 含义 |
|---|---|
| `cv_metrics.csv` | 各模型嵌套交叉验证的 MAE、RMSE、R²、Spearman |
| `fold_metrics.csv` | 每一折的指标，用于检查方差和稳定性 |
| `oof_predictions.csv` | 每行实验的 out-of-fold 预测 |
| `best_model.joblib` | 按 OOF MAE 选出的最终模型与数据 schema |
| `models/*.joblib` | 每一种成功训练的模型 |
| `partial_dependence.csv/.png` | 单变量边际预测关系；是关联而非因果效应 |
| `run_summary.json` | 特征、目标、最佳参数和跳过模型等运行元数据 |

## 4. 用模型推荐下一批实验

候选 CSV 需要包含训练时的全部特征，不需要 `validation_loss`。程序按预测 loss 从低到高排序：

```bash
distill-hparam-suggest \
  --model outputs/example/best_model.joblib \
  --candidates data/example_candidates.synthetic.csv \
  --top-n 20 \
  --output outputs/example/top20_candidates.csv
```

推荐结果适合用于挑选**下一批要实际运行的实验**，不应直接当成最终最优参数。最好把预算分成两部分：大部分测试模型预测最优附近，小部分保留给尚未充分覆盖的参数空间，以降低模型外推失败的风险。

## 5. 实验设计建议

12 维空间很容易出现“看似大量、实际覆盖很差”的情况。建议：

1. 用随机采样、Latin hypercube 或 Sobol 序列覆盖参数空间，不要只做单因素网格。
2. 对重点配置运行 2–3 个 seed，并共享 `experiment_id`。
3. 训练/验证数据版本、teacher/student checkpoint、训练 token 数和学习率等必须固定；若不能固定，应作为特征记录。
4. 先看 `fold_metrics.csv`。如果不同折的 MAE 波动很大或 R² 为负，说明数据量、覆盖范围或特征信息仍不足，不宜相信推荐结果。
5. PDP 在强交互下可能掩盖条件关系。这里的树模型能学习跨领域交互，但如果需要严谨解释，可进一步使用二维 PDP/SHAP，并用新实验验证观察到的规律。

## 6. 测试

```bash
pytest -q
```

重新生成示例数据（会覆盖指定输出文件）：

```bash
python scripts/generate_example_data.py \
  --train-output data/example_experiments.synthetic.csv \
  --candidate-output data/example_candidates.synthetic.csv
```
