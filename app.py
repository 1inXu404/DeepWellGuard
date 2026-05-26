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
    N_FEATURES,
    N_CLASSES as NUM_CLASSES,
    STRIDE as STEP_SIZE,
    RETAINED_CLASSES as TARGET_CLASSES,
    WINDOW_SIZE,
)
from src.models.ablation import AblationCNNLSTMAttention, get_ablation_config
from src.models.bilstm import BiLSTMModel
from src.models.cnn import CNNModel
from src.models.cnn_lstm_attention import CNNLSTMAttention
from src.models.unilstm import UniLSTMModel
from ThreeWToolkit.utils.data_utils import default_data_processing

app = Flask(__name__)
BASE_DIR = Path(__file__).parent
app.config['UPLOAD_FOLDER'] = str(Path(__file__).parent / 'uploads')
app.config['CHECKPOINT_ROOTS'] = [
    str(BASE_DIR / 'results' / 'runs'),
    str(BASE_DIR / 'results' / 'models'),
    str(BASE_DIR / 'results' / 'ablation'),
]
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 从配置加载

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES_CN = {
    0: "正常运行", 1: "含水率突增", 3: "严重段塞流", 4: "流动不稳定",
    5: "产能快速下降", 6: "油嘴快速堵塞", 9: "服务管线水合物形成"
}
CLASS_NAMES = [CLASS_NAMES_CN[c] for c in TARGET_CLASSES]

MODEL_SPECS = {
    "cnn_lstm_attention": {
        "name": "CNN-LSTM-Attention 改进模型",
        "factory": CNNLSTMAttention,
        "badge": "推荐",
    },
    "bilstm": {
        "name": "Bi-LSTM 基线模型",
        "factory": BiLSTMModel,
        "badge": "基线",
    },
    "unilstm": {
        "name": "Uni-LSTM 基线模型",
        "factory": UniLSTMModel,
        "badge": "基线",
    },
    "cnn": {
        "name": "CNN 基线模型",
        "factory": CNNModel,
        "badge": "基线",
    },
    "ablation_cnn_lstm": {
        "name": "CNN-LSTM 消融模型",
        "factory": lambda: AblationCNNLSTMAttention(get_ablation_config("cnn_lstm")),
        "badge": "消融",
    },
    "ablation_lstm_channel_attention": {
        "name": "LSTM-Channel-Attention 消融模型",
        "factory": lambda: AblationCNNLSTMAttention(
            get_ablation_config("lstm_channel_attention")
        ),
        "badge": "消融",
    },
}


def infer_model_key(path: Path) -> str | None:
    """Infer model architecture from a checkpoint path or file name."""
    text = path.as_posix().lower()
    parts = [part.lower() for part in path.parts]

    if "cnn_lstm_attention" in text or "cnnlstmattention" in text:
        return "cnn_lstm_attention"
    if "lstm_channel_attention" in text:
        return "ablation_lstm_channel_attention"
    if "cnn_lstm" in text:
        return "ablation_cnn_lstm"
    if "bi_lstm" in text or "bilstm" in text or "bilstmmodel" in text:
        return "bilstm"
    if "uni_lstm" in text or "unilstm" in text or "unilstmmodel" in text:
        return "unilstm"
    if path.stem.lower() == "lstmmodel":
        return "unilstm"
    if "cnnmodel" in text or ("cnn" in parts and "cnn_lstm" not in text):
        return "cnn"
    return None


def checkpoint_roots() -> list[Path]:
    return [Path(root).resolve() for root in app.config['CHECKPOINT_ROOTS']]


def resolve_checkpoint_path(checkpoint_filename: str) -> Path:
    """Resolve a selected checkpoint while keeping it inside known roots."""
    candidates = []
    raw_path = Path(checkpoint_filename)

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(BASE_DIR / raw_path)
        for root in checkpoint_roots():
            candidates.append(root / raw_path)

    roots = checkpoint_roots()
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_file() and any(path.is_relative_to(root) for root in roots):
            return path

    raise FileNotFoundError(f"找不到权重文件: {checkpoint_filename}")


def load_state_dict(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError("权重文件格式不受支持")

    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }


def checkpoint_entry(pt_file: Path) -> dict:
    model_key = infer_model_key(pt_file)
    spec = MODEL_SPECS.get(model_key, {})
    rel_path = pt_file.relative_to(BASE_DIR).as_posix()
    updated_at = pt_file.stat().st_mtime
    return {
        "path": rel_path,
        "label": f"{spec.get('name', '未知模型')} · {rel_path}",
        "model_key": model_key or "unknown",
        "model_name": spec.get("name", "未知模型"),
        "badge": spec.get("badge", "需确认"),
        "updated_at": updated_at,
        "updated_label": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(updated_at)),
        "recommended": model_key == "cnn_lstm_attention",
    }


@app.route('/get_checkpoints', methods=['GET'])
def get_checkpoints():
    entries = []
    seen = set()

    for root in checkpoint_roots():
        if not root.exists():
            continue
        for pt_file in root.rglob("*.pt"):
            resolved = pt_file.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            entries.append(checkpoint_entry(resolved))

    entries.sort(
        key=lambda item: (
            0 if item["recommended"] else 1,
            -item["updated_at"],
            item["path"],
        )
    )
    return Response(
        json.dumps(entries, ensure_ascii=False),
        mimetype='application/json',
    )


def load_model(checkpoint_filename):
    weight_path = resolve_checkpoint_path(checkpoint_filename)
    model_key = infer_model_key(weight_path)
    if model_key not in MODEL_SPECS:
        raise ValueError(
            "无法识别权重对应的模型结构，请将权重放在 "
            "results/runs/cnn_lstm_attention、results/runs/bi_lstm、"
            "results/runs/uni_lstm 或 results/runs/cnn 等目录中"
        )

    spec = MODEL_SPECS[model_key]
    model = spec["factory"]().to(device)
    model.load_state_dict(load_state_dict(weight_path))
    model.eval()
    return model, {
        "checkpoint": weight_path.relative_to(BASE_DIR).as_posix(),
        "model_key": model_key,
        "model_name": spec["name"],
        "device": str(device),
    }


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
                model, model_meta = load_model(checkpoint_file)
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
            if len(available_cols) != N_FEATURES:
                yield json.dumps({
                    "type": "error",
                    "content": (
                        f"特征列数量不匹配: 当前 {len(available_cols)} 列，"
                        f"模型需要 {N_FEATURES} 列"
                    ),
                }, ensure_ascii=False) + "\n"
                return

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

            X_tensor = torch.from_numpy(np.asarray(X_windows, dtype=np.float32))
            # 转置以匹配 CNNLSTMAttention 模型输入维度 (batch, channels, length) = (batch, 22, 120)
            X_tensor = X_tensor.permute(0, 2, 1)

            yield json.dumps({
                "type": "init",
                "total_windows": len(X_windows),
                "total_points": len(data_np),
                "class_names": CLASS_NAMES,
                "sensor_cols": available_cols,
                "model": model_meta,
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
                        # 增加休眠时间（由 0.02 秒改为 0.1 秒），让前端流式动画显示得更平稳从容
                        time.sleep(0.1)

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
