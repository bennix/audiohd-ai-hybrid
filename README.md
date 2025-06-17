# 🎵 AI+Traditional Hybrid Audio Enhancement System | AI+传统混合音频增强系统

**English** | [中文](#中文版本)

---

## 🌟 English Version

An advanced audio enhancement solution that combines **Artificial Intelligence Deep Learning** and **Traditional Signal Processing** to provide optimal audio quality improvement for your files.

### ✨ Core Features

#### 🤖 AI-Powered Enhancement
- **Facebook Denoiser**: Real-time speech denoising model developed by Meta
- **SpeechBrain Enhancement**: Advanced speech enhancement deep learning model  
- **RNNoise**: Lightweight RNN real-time denoising model
- **Automatic Model Management**: On-demand downloading and loading of AI models

#### 🔧 Traditional Signal Processing
- **Adaptive Denoising**: STFT-based frequency domain noise suppression
- **Harmonic Enhancement**: HPSS separation and frequency domain enhancement
- **Dynamic Range Optimization**: Multi-band compression and dynamic processing
- **Stereo Width Enhancement**: Mid/Side technique for soundstage expansion

#### 🔀 Hybrid Processing Modes
1. **Traditional Only** - Reliable classic algorithms
2. **AI Only** - Pure deep learning enhancement
3. **AI-First Hybrid** - AI processing → Traditional fine-tuning
4. **Traditional-First Hybrid** - Traditional processing → AI enhancement
5. **Parallel Hybrid** - Simultaneous processing with proportional mixing
6. **Adaptive Hybrid** - Intelligent audio feature analysis for automatic strategy selection

#### 🎛️ Complete Custom Control
- **AI Processing Toggle**: Users have full control over AI model usage
- **AI Model Selection**: Free choice of any AI model
- **Flexible Combinations**: All audio qualities can use any AI model and processing mode
- **Smart Recommendations**: System provides suggestions while users maintain full control

### 🚀 Quick Start

#### System Requirements
- Python 3.8+
- Recommended 8GB+ RAM
- GPU optional (CUDA support for AI acceleration)

#### Installation Steps

1. **Clone Repository**
```bash
git clone https://github.com/bennix/audiohd-ai-hybrid.git
cd audiohd-ai-hybrid
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch System**

**Linux/Mac:**
```bash
chmod +x run_hybrid_demo.sh
./run_hybrid_demo.sh
```

**Windows:**
```cmd
run_hybrid_demo.bat
```

4. **Access Interface**
Open browser and visit: http://localhost:7860

### 🎯 Usage Guide

#### Recommended Configuration Combinations

| Audio Type | AI Processing | Recommended Mode | AI Model | Traditional Level | Mix Ratio | Use Cases |
|-----------|---------------|------------------|----------|------------------|-----------|-----------|
| 🎤 Speech Recording | ✅ | Adaptive Hybrid | Facebook Denoiser | Medium | - | Podcasts, interviews, voice notes |
| 🎵 Music Files | ✅ | Parallel Hybrid | SpeechBrain | Medium | 0.6 | Pop music, vintage restoration |
| 📻 Low Quality Audio | ✅ | AI-First Hybrid | RNNoise | Advanced | - | Phone recordings, compressed audio |
| 🎧 High Quality Audio | 🔄 | Traditional-First Hybrid | SpeechBrain | Basic | - | CD quality, professional recordings |
| 🎼 Professional Audio | 🔄 | Parallel Hybrid | Facebook Denoiser | Advanced | 0.3 | Studio work, mastering |
| 🗣️ Meeting Recording | ✅ | Adaptive Hybrid | RNNoise | Medium | - | Remote meetings, multi-speaker |
| 🎙️ Live Recording | ✅ | AI-First Hybrid | Facebook Denoiser | Medium | - | Game streaming, online teaching |
| 🎶 Classical Music | 🔄 | Traditional-First Hybrid | SpeechBrain | Basic | - | Symphony, chamber music |

**Legend**: ✅=Recommended AI enabled, 🔄=Optional AI enabled, -=Adaptive decision

#### Parameter Details

**🤖 AI Processing Control:**
- **Enable Toggle**: Complete control over AI model usage, traditional-only when disabled
- **Model Selection**: Each AI model has unique characteristics and use cases
- **Preloading**: Pre-load AI models for improved processing speed

**🔄 Processing Modes:**
- **Adaptive Hybrid**: System automatically analyzes audio features for optimal strategy
- **Parallel Hybrid**: AI and traditional process simultaneously, results mixed proportionally
- **Sequential Hybrid**: Two methods applied consecutively for mutual enhancement
- **Single Mode**: AI-only or traditional-only processing

**🤖 AI Model Characteristics:**
- **Facebook Denoiser**: Professional speech denoising, excellent for vocal content
- **SpeechBrain**: Balanced speech enhancement, versatile for various audio types
- **RNNoise**: Lightweight real-time denoising, low resource usage, fast processing

**🔧 Traditional Enhancement Levels:**
- **Basic**: Adaptive denoising only, suitable for light enhancement
- **Medium**: Denoising + harmonic enhancement, balanced effect and speed
- **Advanced**: Full processing (denoising+harmonic+dynamic+stereo), best results

### 📊 Technical Architecture

```
Input Audio
    ↓
Audio Feature Analysis
    ↓
User Control Selection
    ↓
┌─────────────────┬─────────────────┐
│   AI Branch      │ Traditional Branch│
│ (Optional Enable)│  (Always Available)│
│                 │                 │
│ • Facebook      │ • Adaptive      │
│   Denoiser      │   Denoising     │
│ • SpeechBrain   │ • Harmonic      │
│ • RNNoise       │   Enhancement   │
│                 │ • Dynamic       │
│                 │   Processing    │
│                 │ • Stereo        │
│                 │   Enhancement   │
└─────────────────┴─────────────────┘
    ↓
Hybrid Strategy Processing
    ↓
Quality Check & Normalization
    ↓
Enhanced Audio Output
```

### 🧠 Intelligent Features

#### Complete User Control
- **AI Toggle**: User decides whether to enable AI processing
- **Model Selection**: Free choice of any AI model
- **Mode Selection**: All audio qualities support any processing mode
- **Parameter Adjustment**: Flexible control of mix ratios and enhancement levels

#### Adaptive Processing Decision
System automatically analyzes audio:
- **Noise Level** - Select appropriate denoising strategy
- **Dynamic Range** - Determine compression processing intensity
- **Spectral Features** - Optimize frequency domain enhancement parameters
- **Voice Activity** - Distinguish between speech and music content

### 📈 Performance Metrics

| Processing Mode | CPU Usage | Memory | Speed | Quality Improvement | Use Cases |
|----------------|-----------|---------|--------|-------------------|-----------|
| Traditional Only | Low | Low | Fast | Good | High-quality audio, fast processing |
| AI Only | Medium-High | Medium | Medium | Excellent | Low-quality audio, professional enhancement |
| Hybrid Mode | Medium | Medium | Medium-Fast | Best | General scenarios, optimal results |
| Adaptive | Dynamic | Medium | Medium | Intelligent | Batch processing, automated mode |

### 🔧 Advanced Configuration

#### Environment Variables
```bash
# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Set model cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# Set parallel processing threads
export OMP_NUM_THREADS=4
```

### 🐛 Troubleshooting

#### Common Issues

**Q: AI model loading failed?**
A: Check network connection, ensure access to HuggingFace Hub. Set proxy or use offline models.

**Q: GPU unavailable?**
A: Check CUDA installation, run `python -c "import torch; print(torch.cuda.is_available())"`

**Q: Processing results have noise?**  
A: Try adjusting mix ratios or use different AI model combinations. For high-quality audio, reduce AI weight.

**Q: Out of memory?**
A: Close unused AI models or adjust processing chunk size.

### 📝 Changelog

#### v2.1.0 (Latest)
- 🎛️ Added complete AI processing custom control
- ✅ High-quality audio can now optionally use AI models
- 🔄 Users can freely enable/disable AI processing
- 📊 More detailed recommendation configuration table
- 🎯 More flexible parameter control options

#### v2.0.0
- ✨ Added AI model support (Facebook Denoiser, SpeechBrain, RNNoise)
- 🔀 Implemented six hybrid processing modes
- 🧠 Added adaptive intelligent processing
- 📊 Optimized audio feature analysis

### 📞 Support & Contribution

- **Bug Reports**: Submit via GitHub Issues
- **Feature Requests**: Welcome to discuss new feature requirements
- **Code Contributions**: Fork project and submit Pull Requests

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

---

<div align="center">

**🎵 Give every audio the best quality it deserves!**

[⭐ Star Project](https://github.com/bennix/audiohd-ai-hybrid) | [🐛 Report Issue](https://github.com/bennix/audiohd-ai-hybrid/issues) | [💡 Feature Request](https://github.com/bennix/audiohd-ai-hybrid/issues)

</div>

---

## 中文版本

一个结合**人工智能深度学习**和**传统信号处理**的高级音频增强解决方案，为您的音频提供最佳的质量提升效果。

## ✨ 核心特性

### 🤖 AI智能增强
- **Facebook Denoiser**: Meta开发的实时语音降噪模型
- **SpeechBrain Enhancement**: 先进的语音增强深度学习模型  
- **RNNoise**: 轻量级RNN实时降噪模型
- **自动模型管理**: 按需下载和加载AI模型

### 🔧 传统信号处理
- **自适应降噪**: 基于STFT的频域噪声抑制
- **谐波增强**: HPSS分离和频率域增强
- **动态范围优化**: 多频段压缩和动态处理
- **立体声宽度增强**: Mid/Side技术扩展声场

### 🔀 混合处理模式
1. **仅传统处理** - 稳定可靠的经典算法
2. **仅AI处理** - 纯深度学习增强
3. **AI优先混合** - AI处理 → 传统精调
4. **传统优先混合** - 传统处理 → AI增强
5. **并行混合** - 同时处理后按比例混合
6. **自适应混合** - 智能分析音频特征自动选择最佳策略

### 🎛️ 完全自定义控制
- **AI处理开关**: 用户可以完全控制是否使用AI模型
- **AI模型选择**: 自由选择使用哪个AI模型
- **灵活组合**: 所有音频质量都可以选择任意AI模型和处理模式
- **智能推荐**: 系统提供推荐设置，但用户拥有完全控制权

## 🚀 快速开始

### 环境要求
- Python 3.8+
- 推荐8GB+内存
- GPU可选(CUDA支持，加速AI处理)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd audiohd
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **启动系统**

**Linux/Mac:**
```bash
chmod +x run_hybrid_demo.sh
./run_hybrid_demo.sh
```

**Windows:**
```cmd
run_hybrid_demo.bat
```

4. **访问界面**
打开浏览器访问: http://localhost:7860

## 🎯 使用指南

### 推荐配置组合

| 音频类型 | AI处理 | 推荐模式 | AI模型 | 传统级别 | 混合比例 | 适用场景 |
|---------|--------|---------|--------|----------|----------|----------|
| 🎤 语音录音 | ✅ | 自适应混合 | Facebook Denoiser | Medium | - | 播客、采访、语音笔记 |
| 🎵 音乐文件 | ✅ | 并行混合 | SpeechBrain | Medium | 0.6 | 流行音乐、老歌修复 |
| 📻 低质量音频 | ✅ | AI优先混合 | RNNoise | Advanced | - | 电话录音、压缩音频 |
| 🎧 高质量音频 | 🔄 | 传统优先混合 | SpeechBrain | Basic | - | CD音质、专业录音 |
| 🎼 专业音频 | 🔄 | 并行混合 | Facebook Denoiser | Advanced | 0.3 | 录音棚作品、母带处理 |
| 🗣️ 会议录音 | ✅ | 自适应混合 | RNNoise | Medium | - | 远程会议、多人对话 |
| 🎙️ 直播录音 | ✅ | AI优先混合 | Facebook Denoiser | Medium | - | 游戏直播、在线教学 |
| 🎶 古典音乐 | 🔄 | 传统优先混合 | SpeechBrain | Basic | - | 交响乐、室内乐 |

**说明**: ✅=推荐开启AI, 🔄=可选择开启AI, -=自适应决定

### 参数详细说明

**🤖 AI处理控制:**
- **启用开关**: 完全控制是否使用AI模型，关闭后只使用传统算法
- **模型选择**: 每个AI模型都有不同的特色和适用场景
- **预加载**: 可以提前加载AI模型以提高处理速度

**🔄 处理模式:**
- **自适应混合**: 系统自动分析音频特征，选择最佳处理策略
- **并行混合**: AI和传统同时处理，结果按比例混合
- **顺序混合**: 先后使用两种方法，相互补强
- **单一模式**: 仅使用AI或仅使用传统方法

**🤖 AI模型特点:**
- **Facebook Denoiser**: 专业语音降噪，适合人声内容，效果显著
- **SpeechBrain**: 平衡的语音增强，适合多种音频类型，通用性强
- **RNNoise**: 轻量级实时降噪，资源占用少，速度快，适合实时处理

**🔧 传统增强级别:**
- **Basic**: 仅自适应降噪，适合轻微增强
- **Medium**: 降噪 + 谐波增强，平衡效果和速度
- **Advanced**: 全套处理(降噪+谐波+动态+立体声)，效果最佳

**⚖️ 混合比例:**
- **1.0**: 完全AI处理，最大化AI效果
- **0.7-0.8**: AI为主，传统辅助，适合AI效果好的场景
- **0.5**: 平衡混合，结合两者优势
- **0.2-0.3**: 传统为主，AI辅助，适合高质量音频
- **0.0**: 完全传统处理，最稳定的效果

## 📊 技术架构

```
输入音频
    ↓
音频特征分析
    ↓
用户控制选择
    ↓
┌─────────────────┬─────────────────┐
│   AI处理分支     │   传统处理分支   │
│  (可选择启用)    │   (始终可用)     │
│                │                │
│ • Facebook      │ • 自适应降噪    │
│   Denoiser     │ • 谐波增强      │
│ • SpeechBrain  │ • 动态处理      │
│ • RNNoise      │ • 立体声增强    │
└─────────────────┴─────────────────┘
    ↓
混合策略处理
    ↓
质量检查与标准化
    ↓
输出增强音频
```

## 🧠 智能特性

### 用户完全控制
- **AI开关**: 用户决定是否启用AI处理
- **模型选择**: 自由选择任意AI模型
- **模式选择**: 支持所有音频质量使用任意处理模式
- **参数调节**: 灵活调整混合比例和增强级别

### 自适应处理决策
系统自动分析音频的：
- **噪声水平** - 选择合适的降噪策略
- **动态范围** - 决定压缩处理强度
- **频谱特征** - 优化频域增强参数
- **语音活动** - 区分语音和音乐内容

### 实时性能优化
- **模型缓存**: 避免重复加载AI模型
- **预加载功能**: 用户可以预先加载常用模型
- **并行处理**: 立体声双声道同时处理
- **内存管理**: 大文件分块处理
- **GPU加速**: 自动检测并使用CUDA

## 📈 性能表现

| 处理模式 | CPU使用率 | 内存占用 | 处理速度 | 质量提升 | 适用场景 |
|---------|----------|----------|----------|----------|----------|
| 仅传统 | 低 | 低 | 快 | 良好 | 高质量音频、快速处理 |
| 仅AI | 中高 | 中 | 中 | 优秀 | 低质量音频、专业增强 |
| 混合模式 | 中 | 中 | 中快 | 最佳 | 通用场景、最佳效果 |
| 自适应 | 动态 | 中 | 中 | 智能 | 批量处理、懒人模式 |

## 🔧 高级配置

### 环境变量
```bash
# 设置GPU设备
export CUDA_VISIBLE_DEVICES=0

# 设置模型缓存目录
export TRANSFORMERS_CACHE=/path/to/cache

# 设置并行处理线程数
export OMP_NUM_THREADS=4
```

### 自定义模型
可以通过修改 `ai_models.py` 添加更多AI模型支持。

### 用户使用技巧

**🎯 针对不同音频质量的建议:**

1. **超高质量音频(录音棚级别)**:
   - 可以关闭AI处理，仅使用传统Basic级别
   - 或使用传统优先混合，AI权重设为0.2-0.3

2. **高质量音频(CD音质)**:
   - 推荐传统优先混合 + SpeechBrain
   - AI权重可设置为0.3-0.5

3. **中等质量音频(MP3/AAC)**:
   - 推荐并行混合，AI权重0.5-0.7
   - 可以尝试不同AI模型对比效果

4. **低质量音频(电话/压缩)**:
   - 推荐AI优先混合或仅AI处理
   - RNNoise和Facebook Denoiser效果较好

## 🐛 故障排除

### 常见问题

**Q: AI模型加载失败？**
A: 检查网络连接，确保可以访问HuggingFace Hub。可以设置代理或使用离线模型。

**Q: GPU不可用？**
A: 检查CUDA安装，运行 `python -c "import torch; print(torch.cuda.is_available())"`

**Q: 处理结果有噪音？**  
A: 尝试调整混合比例，或使用不同的AI模型组合。高质量音频可以降低AI权重。

**Q: 内存不足？**
A: 关闭不使用的AI模型，或调整处理块大小。

**Q: 高质量音频效果不明显？**
A: 高质量音频建议使用传统优先混合，或降低AI权重，避免过度处理。

### 性能优化建议

1. **首次使用**: 预先加载常用AI模型
2. **批量处理**: 保持模型在内存中连续处理
3. **GPU加速**: 确保PyTorch CUDA版本正确安装
4. **内存管理**: 处理大文件后重启应用释放内存
5. **参数调优**: 根据音频质量选择合适的AI权重

## 📝 更新日志

### v2.1.0 (最新)
- 🎛️ 新增AI处理完全自定义控制
- ✅ 高质量音频也可选择使用AI模型
- 🔄 用户可自由开启/关闭AI处理
- 📊 更详细的推荐配置表格
- 🎯 更灵活的参数控制选项

### v2.0.0
- ✨ 新增AI模型支持(Facebook Denoiser, SpeechBrain, RNNoise)
- 🔀 实现六种混合处理模式
- 🧠 添加自适应智能处理
- 📊 优化音频特征分析
- 🎨 全新用户界面设计

### v1.0.0
- 🔧 传统信号处理功能
- 🎵 基础音频增强
- 📱 Gradio Web界面

## 📞 支持与贡献

- **Bug报告**: 请在GitHub Issues中提交
- **功能建议**: 欢迎讨论新功能需求
- **代码贡献**: Fork项目并提交Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

---

<div align="center">

**🎵 让每一段音频都拥有最佳音质！**

[⭐ Star项目](.) | [🐛 报告问题](.) | [�� 功能建议](.)

</div> 