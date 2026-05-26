# 本地训练结果目录

这个目录用于保存本地训练、评估和可视化产物。除本说明文件外，`results/` 下的内容默认被 `.gitignore` 忽略，不随代码提交。

如果需要复现实验结果，请先完成预处理和训练，再运行评估脚本：

```powershell
python scripts/compare.py
python scripts/detailed_metrics.py
```

## 目录结构

- `summary/`: 主模型对比表和对比柱状图。
- `runs/`: 四个主模型的完整产物，每个模型一个文件夹。
- `ablation/`: 消融实验汇总、F1 对比图和各消融变体产物。
- `cache/`: 训练和测试缓存数据。
- `notebook_checkpoints/`: Jupyter 自动生成的 checkpoint 文件，单独归档，避免混在正式结果里。

## 混淆矩阵

混淆矩阵图片包括：

- `confusion_normalized.png`: 按真实类别归一化后的混淆矩阵。
- `confusion_counts.png`: 计数形式的混淆矩阵。

图中的类别名使用映射后的模型输出序号 `class0` 到 `class6`，对应原始保留类别 `[0, 1, 3, 4, 5, 6, 9]`。图片不显示标题，便于在论文、报告或幻灯片中统一排版。

## 主模型结果

| 模型 | Accuracy | Weighted F1 | Macro F1 | 位置 |
| --- | ---: | ---: | ---: | --- |
| CNN-LSTM-Attention | 0.9360 | 0.9375 | 0.9297 | `runs/cnn_lstm_attention/` |
| Bi-LSTM | 0.9318 | 0.9328 | 0.9259 | `runs/bi_lstm/` |
| Uni-LSTM | 0.9273 | 0.9281 | 0.9197 | `runs/uni_lstm/` |
| CNN | 0.9200 | 0.9190 | 0.9108 | `runs/cnn/` |

每个主模型目录内统一包含：

- `model.pt`
- `training_history.csv`
- `predictions.npz`
- `training_curves.png`
- `classification_report.txt`
- `per_class_metrics.csv`
- `confusion_normalized.png`
- `confusion_counts.png`
- `roc_curve.png`

## 消融实验

| 变体 | Accuracy | Weighted F1 | Macro F1 | Best Val Acc | Epochs | 位置 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| lstm_channel_attention | 0.9311 | 0.9315 | 0.9204 | 0.9629 | 35 | `ablation/variants/lstm_channel_attention/` |
| cnn_lstm | 0.9227 | 0.9229 | 0.9160 | 0.9491 | 17 | `ablation/variants/cnn_lstm/` |

正式对比请优先看 `summary/model_comparison.csv` 和 `summary/model_comparison_bar.png`。
