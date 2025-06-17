#!/bin/bash

echo "🎵 启动AI+传统混合音频增强系统..."
echo "========================================"

# 检查Python版本
python_version=$(python3 --version 2>&1)
echo "📋 Python版本: $python_version"

# 检查是否安装了必要的包
echo "📦 检查依赖包..."
if ! python3 -c "import gradio" 2>/dev/null; then
    echo "❌ 缺少gradio包，正在安装..."
    pip3 install -r requirements.txt
fi

# 检查GPU支持
if python3 -c "import torch; print('✅ GPU可用' if torch.cuda.is_available() else '⚠️ 仅CPU模式')" 2>/dev/null; then
    echo "🔥 PyTorch GPU检查完成"
else
    echo "⚠️ PyTorch未安装或有问题"
fi

echo ""
echo "🚀 启动混合音频增强系统..."
echo "🌐 系统将在 http://localhost:7860 启动"
echo "📱 支持的功能:"
echo "   • 🤖 AI模型: Facebook Denoiser, SpeechBrain, RNNoise"
echo "   • 🔧 传统处理: 自适应降噪, 谐波增强, 动态范围优化"
echo "   • 🔀 混合模式: AI优先, 传统优先, 并行混合, 自适应混合"
echo ""
echo "⏳ 正在启动服务器..."

# 启动应用
python3 app_hybrid.py 