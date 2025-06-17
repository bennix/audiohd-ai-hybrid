@echo off
chcp 65001 > nul

echo 🎵 启动AI音频超分辨率演示...
echo ================================

:: 检查Python是否安装
python --version > nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: Python 未安装
    echo 请先安装 Python 3.8 或更高版本
    pause
    exit /b 1
)

:: 检查pip是否安装
pip --version > nul 2>&1
if errorlevel 1 (
    echo ❌ 错误: pip 未安装
    echo 请先安装 pip
    pause
    exit /b 1
)

:: 创建虚拟环境（可选）
if not exist "venv" (
    echo 📦 创建虚拟环境...
    python -m venv venv
)

:: 激活虚拟环境
echo 🔄 激活虚拟环境...
call venv\Scripts\activate.bat

:: 安装依赖
echo 📥 安装依赖包...
pip install -r requirements.txt

:: 检查依赖是否安装成功
echo 🔍 检查依赖安装...
python -c "import gradio, torch, librosa, soundfile, scipy, numpy" 2>nul
if errorlevel 1 (
    echo ❌ 依赖安装失败，请检查错误信息
    pause
    exit /b 1
) else (
    echo ✅ 所有依赖安装成功
)

:: 启动应用
echo 🚀 启动应用...
echo ================================
echo 📱 访问地址: http://localhost:7860
echo 🛑 按 Ctrl+C 停止应用
echo ================================

python app.py

pause 