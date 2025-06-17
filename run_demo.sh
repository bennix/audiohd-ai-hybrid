#!/bin/bash

echo "🎵 启动AI音频超分辨率演示..."
echo "================================"

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: Python3 未安装"
    echo "请先安装 Python 3.8 或更高版本"
    exit 1
fi

# 检查pip是否安装
if ! command -v pip3 &> /dev/null; then
    echo "❌ 错误: pip3 未安装"
    echo "请先安装 pip3"
    exit 1
fi

# 创建虚拟环境（可选）
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔄 激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "📥 安装依赖包..."
pip install -r requirements.txt

# 检查依赖是否安装成功
echo "🔍 检查依赖安装..."
python3 -c "import gradio, torch, librosa, soundfile, scipy, numpy" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ 所有依赖安装成功"
else
    echo "❌ 依赖安装失败，请检查错误信息"
    exit 1
fi

# 启动应用
echo "🚀 启动应用..."
echo "================================"
echo "📱 访问地址: http://localhost:7860"
echo "🛑 按 Ctrl+C 停止应用"
echo "================================"

python3 app.py 