import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from flask import (
    Flask,
    Response,
    render_template,
    request,
    stream_with_context,
)
from werkzeug.utils import secure_filename

from src.utils.config import (
    DOWNSAMPLE_RATE,
    N_CLASSES as NUM_CLASSES,
    STRIDE as STEP_SIZE,
    RETAINED_CLASSES as TARGET_CLASSES,
    WINDOW_SIZE,
)
from src.models.cnn_lstm_attention import CNNLSTMAttention
from src.models.lstm import LSTMModel
from ThreeWToolkit.utils.data_utils import default_data_processing

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(Path(__file__).parent / 'uploads')
app.config['CHECKPOINT_FOLDER'] = str(Path(__file__).parent / 'results' / 'models')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 从配置加载

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES_CN = {
    0: "正常运行", 1: "含水率突增", 3: "严重段塞流", 4: "流动不稳定",
    5: "产能快速下降", 6: "油嘴快速堵塞", 9: "服务管线水合物形成"
}
CLASS_NAMES = [CLASS_NAMES_CN[c] for c in TARGET_CLASSES]


@app.route('/get_checkpoints', methods=['GET'])
def get_checkpoints():
    folder = Path(app.config['CHECKPOINT_FOLDER'])
    if not folder.exists():
        return json.dumps([])

    files = []
    # 查找所有的 .pt 权重文件，包括子文件夹
    for pt_file in folder.rglob("*.pt"):
        # 存下相对路径，例如 "20260517_115543/cnnlstmattention.pt"
        rel_path = pt_file.relative_to(folder).as_posix()
        files.append(rel_path)

    files.sort(reverse=True)
    return json.dumps(files)


def load_model(checkpoint_filename):
    weight_path = os.path.join(
        app.config['CHECKPOINT_FOLDER'],
        checkpoint_filename)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到权重文件: {weight_path}")

    # 获取训练时使用的传感器列 (3W数据集中排除class和state的剩余列，共22个)
    # 因为 config 里没有 load_feature_columns，我们硬编码或者在此方法中返回 None 让后面处理
    # 实际上预处理会自动丢弃非信号列，所以不需要返回 feature_columns
    
    if "lstmmodel" in checkpoint_filename.lower():
        model = LSTMModel().to(device)
    else:
        model = CNNLSTMAttention().to(device)

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model, None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict_stream', methods=['POST'])
def predict_stream():
    if 'file' not in request.files:
        return Response(
            json.dumps({"type": "error", "content": "没有上传文件"}) + "\n",
            mimetype='application/x-ndjson')
    file = request.files['file']
    checkpoint_file = request.form.get('checkpoint_file')

    if file.filename == '':
        return Response(json.dumps({"type": "error",
                                    "content": "未选择任何数据文件"}) + "\n",
                        mimetype='application/x-ndjson')
    if not checkpoint_file:
        return Response(json.dumps({"type": "error",
                                    "content": "未选择权重文件"}) + "\n",
                        mimetype='application/x-ndjson')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    def generate():
        try:
            yield json.dumps({
                "type": "message",
                "content": "正在加载模型与权重...",
            }) + "\n"
            try:
                model, _ = load_model(checkpoint_file)
            except Exception as e:
                yield json.dumps({
                    "type": "error",
                    "content": f"加载模型失败: {str(e)}",
                }) + "\n"
                return

            yield json.dumps({
                "type": "message",
                "content": "正在预处理数据...",
            }) + "\n"
            df = pd.read_parquet(filepath, engine="pyarrow")

            # 降采样
            if DOWNSAMPLE_RATE > 1:
                df = df.iloc[::DOWNSAMPLE_RATE].reset_index(drop=True)

            # 分离信号和标签
            target_column = "class"
            signals = df.drop(columns=[target_column], errors='ignore')
            labels = (
                df[[target_column]]
                if target_column in df.columns
                else pd.DataFrame({target_column: [0] * len(df)})
            )

            # 官方预处理流程
            data = {
                "signal": signals,
                "label": labels,
                "file_name": filename
            }
            data = default_data_processing(
                data, fillna=True,
                target_column=target_column,
                fill_target_value=0
            )
            signals = data["signal"]

            # 提取可用特征列 (去除非信号列)
            available_cols = [c for c in signals.columns if c not in ("class", "state")]
            display_values = {col: signals[col].fillna(
                0).tolist() for col in available_cols}
            data_np = signals[available_cols].values.astype(np.float32)

            # 创建滑动窗口
            X_windows = []
            for i in range(0, len(data_np) - WINDOW_SIZE + 1, STEP_SIZE):
                X_windows.append(data_np[i: i + WINDOW_SIZE])

            if not X_windows:
                yield json.dumps({
                    "type": "error",
                    "content": (
                        f"数据长度不足 "
                        f"(至少需要 {WINDOW_SIZE} 步)"
                    ),
                }) + "\n"
                return

            X_tensor = torch.tensor(X_windows, dtype=torch.float32)
            # 转置以匹配 CNNLSTMAttention 模型输入维度 (batch, channels, length) = (batch, 22, 120)
            X_tensor = X_tensor.permute(0, 2, 1)

            yield json.dumps({
                "type": "init",
                "total_windows": len(X_windows),
                "total_points": len(data_np),
                "class_names": CLASS_NAMES,
                "sensor_cols": available_cols
            }) + "\n"

            states = np.zeros(len(data_np), dtype=int)
            batch_size = 64

            with torch.no_grad():
                for i in range(0, X_tensor.size(0), batch_size):
                    batch_x = X_tensor[i:i + batch_size].to(device)
                    outputs = model(batch_x)
                    probs = F.softmax(outputs, dim=1).cpu().numpy()
                    preds = torch.max(outputs.data, 1)[1].cpu().numpy()

                    for b in range(len(preds)):
                        win_idx = i + b
                        start_idx = win_idx * STEP_SIZE
                        end_idx = start_idx + WINDOW_SIZE

                        # 将模型输出索引映射回原始类别
                        state_val = int(TARGET_CLASSES[preds[b]])
                        state_idx = int(preds[b])
                        prob_val = probs[b].tolist()

                        if win_idx == 0:
                            vals = {
                                col: display_values[col][0: WINDOW_SIZE]
                                for col in available_cols}
                        else:
                            vals = {
                                col: display_values[col][
                                    max(0, end_idx - STEP_SIZE):end_idx
                                ]
                                for col in available_cols
                            }

                        if state_val > 0:
                            states[start_idx:end_idx] = np.maximum(
                                states[start_idx:end_idx], state_val)

                        chunk = {
                            "type": "data",
                            "win_idx": win_idx,
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "new_values": vals,
                            "state": state_val,
                            "state_idx": state_idx,
                            "probs": prob_val
                        }
                        yield json.dumps(chunk) + "\n"
                        time.sleep(0.02)

            abnormal_preds = [p for p in states if p != 0]
            if abnormal_preds:
                dominant = max(set(abnormal_preds), key=abnormal_preds.count)
                status_text = (
                    f"总览: 检测到主要异常 "
                    f"[{CLASS_NAMES_CN.get(dominant, '未知')}]"
                )
            else:
                status_text = "总览: 文件整体运行正常"
                dominant = 0

            yield json.dumps({
                "type": "done",
                "status": status_text,
                "dominant_class": int(dominant)
            }) + "\n"

        except Exception as e:
            yield json.dumps({
                "type": "error",
                "content": f"运行异常: {str(e)}",
            }) + "\n"

    return Response(
        stream_with_context(
            generate()),
        mimetype='application/x-ndjson')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
