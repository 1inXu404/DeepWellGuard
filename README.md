# DeepWellGuard: Oil Well Anomaly Detection System

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

本项目为**《基于 CNN-LSTM-Attention 的油井不良事件检测系统设计》**的官方代码实现。
针对真实工业场景下的油井多传感器时序数据（基于 3W Dataset），提出了一种融合多尺度卷积、挤压激励（SE）通道注意力与时间自注意力的深度学习架构，实现了工业级流式传感器数据的实时故障诊断。

---

## 🌟 核心亮点 (Key Features)

- 🧠 **融合式混合架构**：突破传统单一模型限制，完美融合空间局部特征（CNN）与长时序演进因果关系（LSTM）。
- 🔭 **多尺度与双重注意力机制**：
  - **MSCNN (多尺度卷积)**：并行使用 3、7、11 三种尺寸的卷积核，同时捕捉瞬间突变与缓慢憋压。
  - **SE Block (通道注意力)**：动态评估 22 个传感器的权重，自动静音无效背景噪声。
  - **Multi-Head Temporal Attention (时间注意力)**：在 120 个时间步中自动聚焦最具决定性的异常关键帧。
- ⚡ **工业级流式推理流**：采用降采样、PyArrow 极速读取、向量化预处理与 GPU 批推理技术，实现 Web 端秒级响应。
- 📊 **压倒性的性能表现**：在近 10 万规模的独立测试集上，相比纯 Bi-LSTM 基线，改进模型的 **Macro F1 从 93.71% 暴涨至 95.16%**。

---

## 📈 实验结果 (Results)

| 模型架构 | 准确率 (Accuracy) | 加权 F1 (Weighted F1) | 宏 F1 (Macro F1) |
| :--- | :---: | :---: | :---: |
| 纯 Bi-LSTM 基线 | 94.04% | 94.10% | 93.71% |
| **CNN-LSTM-Attention (改进版)** | **94.94%** | **94.96%** | **95.16%** 🚀 |

> *注：上述结果已排除数据量极少的 2, 7, 8 类别。实验数据切分采用严格的分层随机抽样 (Stratified Random Sampling)，并使用从未参与训练的独立测试集 (Holdout Test Set)。*

训练、评估和可视化产物会输出到 `results/`，包括模型权重、预测文件、训练曲线、ROC 曲线、混淆矩阵、分类报告和汇总表。为避免把大文件与重复生成的实验产物提交到仓库，`results/` 下的训练输出默认被 `.gitignore` 忽略；仅保留 `results/README.md` 作为目录说明。

混淆矩阵显示使用模型映射后的类别序号 `class0` 到 `class6`，对应原始保留类别 `[0, 1, 3, 4, 5, 6, 9]`。生成的混淆矩阵图片不显示标题，便于论文或报告排版。

---

## 📂 项目结构 (Project Structure)

```text
DeepWellGuard/
├── app.py                      # Flask 实时流式监控前端入口
├── scripts/                    # 执行脚本存放区
│   ├── create_demo_file.py     # 自动缝合生成包含所有故障的时序测试文件
│   ├── train_cnn.py            # 训练纯 CNN 基线模型的入口
│   ├── train_unilstm.py        # 训练纯单向 LSTM 基线模型的入口
│   ├── train_bilstm.py         # 训练纯 Bi-LSTM 基线模型的入口
│   ├── train_baseline.py       # 统一基线训练入口 (--model cnn/unilstm/bilstm)
│   ├── train_cnn_lstm_attn.py  # 训练改进版模型的入口
│   ├── train_ablation.py       # CNN-LSTM-Attention 消融实验入口
│   ├── compare.py              # 模型对比与评估画图脚本
│   └── cleanup_models.py       # 自动清理历史废弃权重的脚本
├── src/
│   ├── data/                   # 数据预处理与 DataLoader 模块
│   ├── models/                 # PyTorch 网络架构定义
│   │   ├── bilstm.py
│   │   ├── cnn.py
│   │   ├── cnn_lstm_attention.py
│   │   └── unilstm.py
│   ├── train/                  # 训练循环控制、早停、AMP、评估计算
│   └── utils/                  # 全局超参数与配置文件 (config.py)
├── results/                    # 本地实验输出区 (默认被 Git 忽略，仅保留 README)
├── templates/ & static/        # Web 监控平台 UI 资源
└── requirements.txt            # 项目依赖
```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境安装
```powershell
git clone https://github.com/YourUsername/DeepWellGuard.git
cd DeepWellGuard
pip install -r requirements.txt
```

### 2. 模型训练与评估
确保你已下载 `3w_dataset_2.0.0` 数据集并放置在根目录下，然后依次运行：
```powershell
# 训练纯 CNN / 纯单向 LSTM / 纯 Bi-LSTM 基线与改进模型
python scripts/train_cnn.py --epochs 100 --batch-size 128
python scripts/train_unilstm.py --epochs 100 --batch-size 128
python scripts/train_bilstm.py --epochs 100 --batch-size 128
python scripts/train_cnn_lstm_attn.py --epochs 100 --batch-size 128

# 运行消融实验：LSTM+Attention 与 CNN+LSTM
python scripts/train_ablation.py --epochs 100 --batch-size 128

# 评估、生成对比指标及混淆矩阵
python scripts/compare.py
python scripts/detailed_metrics.py
```

评估脚本会在本地 `results/` 中生成结果文件；这些输出默认不会被 Git 跟踪。如需共享某次实验结果，请手动整理需要保留的表格或图片。

### Linux 服务器全量实验
如需先用最小样本验证完整训练链路：
```bash
python scripts/train_cnn.py --epochs 1 --batch-size 32 --subset 0.01
python scripts/train_unilstm.py --epochs 1 --batch-size 32 --subset 0.01
python scripts/train_bilstm.py --epochs 1 --batch-size 32 --subset 0.01
python scripts/train_cnn_lstm_attn.py --epochs 1 --batch-size 32 --subset 0.01
python scripts/train_ablation.py --epochs 1 --batch-size 32 --subset 0.01
python scripts/compare.py
python scripts/detailed_metrics.py
```

在 Linux 服务器上后台运行全量实验，并在全部任务成功完成后自动关机：
```bash
export RUN_DIR="logs/experiments/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

PYTHONUNBUFFERED=1 nohup bash -c '
set -euo pipefail

if [ ! -f results/cache/fold_train_X.npy ] || [ ! -f results/cache/test_X.npy ]; then
  python scripts/preprocess.py 2>&1 | tee "$RUN_DIR/00_preprocess.log"
fi

python scripts/train_cnn.py --epochs 100 --batch-size 128 --subset 1.0 2>&1 | tee "$RUN_DIR/01_train_cnn_baseline.log"
python scripts/train_unilstm.py --epochs 100 --batch-size 128 --subset 1.0 2>&1 | tee "$RUN_DIR/02_train_unilstm_baseline.log"
python scripts/train_bilstm.py --epochs 100 --batch-size 128 --subset 1.0 2>&1 | tee "$RUN_DIR/03_train_bilstm_baseline.log"
python scripts/train_cnn_lstm_attn.py --epochs 100 --batch-size 128 --subset 1.0 2>&1 | tee "$RUN_DIR/04_train_cnn_lstm_attention.log"
python scripts/train_ablation.py --epochs 100 --batch-size 128 --subset 1.0 2>&1 | tee "$RUN_DIR/05_train_default_ablation.log"
python scripts/compare.py 2>&1 | tee "$RUN_DIR/06_compare_models.log"
python scripts/detailed_metrics.py 2>&1 | tee "$RUN_DIR/07_detailed_metrics.log"

shutdown -h now
' > run_all.out 2>&1 &
```

实时查看运行输出：
```bash
tail -f run_all.out
```

各步骤日志会保存到 `logs/experiments/<timestamp>/`。如果当前用户没有直接关机权限，请使用 `sudo` 启动任务，或将最后一行改为 `sudo -n shutdown -h now` 并配置免密关机权限。

### 3. 启动流式监控大屏
你可以生成一段包含所有故障连续发生的测试流数据，并通过网页上传监控。
```powershell
# 生成包含全类别的长序列测试文件 (存至 uploads/demo_test_sequence.parquet)
python scripts/create_demo_file.py

# 启动 Web 前端界面
python app.py
```
浏览器打开 `http://127.0.0.1:5000`，选择生成的 parquet 测试文件，体验毫秒级流式故障诊断。

---

## 🙏 致谢与引用 (Acknowledgments)
本项目使用的原始传感器数据来源于巴西国家石油公司 (Petrobras) 开源的 **3W Dataset**。
感谢 Petrobras 对工业界开源数据集的巨大贡献！

- **3W 数据集官方 GitHub 仓库**：[https://github.com/petrobras/3W](https://github.com/petrobras/3W)

如果你在研究中使用了本项目或 3W 数据集，请考虑引用原作者的相关文献。

---

## 📄 许可证 (License)
本项目采用 [MIT License](LICENSE) 开源协议。
