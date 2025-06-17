@echo off
chcp 65001 >nul
title AI+传统混合音频增强系统

echo 🎵 启动AI+传统混合音频增强系统...
echo ========================================

REM 检查Python版本
echo 📋 检查Python版本...
python --version 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python未安装或不在PATH中
    pause
    exit /b 1
)

REM 检查依赖包
echo 📦 检查依赖包...
python -c "import gradio" 2>nul
if %errorlevel% neq 0 (
    echo ❌ 缺少gradio包，正在安装...
    pip install -r requirements.txt
)

REM 检查GPU支持
echo 🔥 检查GPU支持...
python -c "import torch; print('✅ GPU可用' if torch.cuda.is_available() else '⚠️ 仅CPU模式')" 2>nul

echo.
echo 🚀 启动混合音频增强系统...
echo 🌐 系统将在 http://localhost:7860 启动
echo 📱 支持的功能:
echo    • 🤖 AI模型: Facebook Denoiser, SpeechBrain, RNNoise
echo    • 🔧 传统处理: 自适应降噪, 谐波增强, 动态范围优化  
echo    • 🔀 混合模式: AI优先, 传统优先, 并行混合, 自适应混合
echo.
echo ⏳ 正在启动服务器...

REM 启动应用
python app_hybrid.py

pause 