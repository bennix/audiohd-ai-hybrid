import gradio as gr
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from typing import Tuple, Optional
import warnings
from hybrid_enhancer import hybrid_enhancer
warnings.filterwarnings("ignore")

def process_audio_hybrid(audio_file, processing_mode, enable_ai, ai_model, enhancement_level, blend_ratio):
    """混合音频处理主函数"""
    if audio_file is None:
        return None, "❌ 请上传音频文件", ""
    
    try:
        print(f"\n🎵 开始处理音频文件: {audio_file}")
        print(f"📊 处理模式: {processing_mode}")
        print(f"🤖 启用AI: {enable_ai}")
        print(f"🤖 AI模型: {ai_model if enable_ai else '未使用'}")
        print(f"🔧 传统增强级别: {enhancement_level}")
        
        # 读取音频文件
        audio, sr = librosa.load(audio_file, sr=None, mono=False)
        print(f"📂 音频信息: {audio.shape}, 采样率: {sr}Hz")
        
        # 根据AI开关调整处理模式
        if not enable_ai:
            if processing_mode in ["ai_only", "ai_then_traditional", "traditional_then_ai", "parallel_blend"]:
                processing_mode = "traditional_only"
                print("⚠️ AI处理已禁用，自动切换到仅传统处理模式")
        
        # 处理立体声/单声道
        if len(audio.shape) > 1 and audio.shape[0] == 2:
            # 立体声：分别处理左右声道
            print("🎧 处理立体声音频...")
            left_channel = audio[0]
            right_channel = audio[1]
            
            # 计算正确的音频时长（使用单个声道的长度）
            audio_duration = len(left_channel) / sr
            
            # 处理左声道
            enhanced_left, metadata_left = hybrid_enhancer.enhance_audio(
                left_channel, sr, processing_mode, ai_model, enhancement_level, blend_ratio
            )
            
            # 处理右声道
            enhanced_right, metadata_right = hybrid_enhancer.enhance_audio(
                right_channel, sr, processing_mode, ai_model, enhancement_level, blend_ratio
            )
            
            # 合并立体声
            enhanced_audio = np.array([enhanced_left, enhanced_right])
            metadata = metadata_left  # 使用左声道的元数据
            
        else:
            # 单声道
            print("🎵 处理单声道音频...")
            if len(audio.shape) > 1:
                audio = audio[0]  # 取第一个声道
            
            # 计算音频时长
            audio_duration = len(audio) / sr
            
            enhanced_audio, metadata = hybrid_enhancer.enhance_audio(
                audio, sr, processing_mode, ai_model, enhancement_level, blend_ratio
            )
        
        # 保存处理后的音频
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            output_path = tmp_file.name
            
            # 根据音频维度保存
            if len(enhanced_audio.shape) > 1:
                # 立体声：转置矩阵以匹配soundfile格式 (time, channels)
                sf.write(output_path, enhanced_audio.T, sr)
            else:
                # 单声道
                sf.write(output_path, enhanced_audio, sr)
        
        # 生成处理报告
        if metadata.get("success", False):
            status_message = f"✅ 处理完成!\n📊 使用方法: {metadata['method_used']}"
            
            # 添加音频特征信息
            features = metadata.get('original_features', {})
            feature_info = f"""
            
📈 音频特征分析:
• 噪声水平: {features.get('noise_level', 0):.3f}
• 动态范围: {features.get('dynamic_range', 0):.3f}  
• 频谱质心: {features.get('spectral_centroid', 0):.0f}Hz
• 零交叉率: {features.get('zero_crossing_rate', 0):.3f}

🔧 实际处理方法:
{metadata.get('actual_method_details', '详情不可用')}
            """
            status_message += feature_info
        else:
            status_message = f"❌ 处理失败: {metadata.get('error', '未知错误')}"
        
        # 生成处理详情 - 使用准确的元数据
        ai_status = "已启用" if enable_ai else "已禁用"
        ai_model_used = metadata.get('ai_model_used', '未使用')
        traditional_used = "是" if metadata.get('traditional_used', False) else "否"
        
        # 使用混合处理器返回的显示文本
        blend_ratio_display = metadata.get('blend_ratio_display', '不适用')
        
        process_details = f"""
🔍 处理详情:
• 原始采样率: {sr}Hz
• 音频时长: {audio_duration:.2f}秒
• 处理模式: {processing_mode}
• AI处理状态: {ai_status}
• AI模型: {ai_model_used}
• 传统处理: {traditional_used}
• 传统增强级别: {enhancement_level}
• 混合比例: {blend_ratio_display}
        """
        
        print("✅ 音频处理完成")
        return output_path, status_message, process_details
        
    except Exception as e:
        error_msg = f"❌ 处理过程中发生错误: {str(e)}"
        print(error_msg)
        return None, error_msg, ""

def load_ai_model_interface(model_name):
    """加载AI模型的界面函数"""
    if not model_name:
        return "❌ 请选择要加载的AI模型"
    
    print(f"⏳ 正在加载AI模型: {model_name}")
    success = hybrid_enhancer.load_ai_model(model_name)
    
    if success:
        return f"✅ AI模型 {model_name} 加载成功!"
    else:
        return f"❌ AI模型 {model_name} 加载失败"

def get_model_info(model_name):
    """获取模型信息"""
    ai_models = hybrid_enhancer.get_ai_models()
    if model_name in ai_models:
        info = ai_models[model_name]
        return f"""
🤖 {info['name']}
📝 {info['description']}
💾 模型大小: {info['size']}
🎯 主要功能: {info['task']}
        """
    return "请选择一个AI模型查看详情"

def update_ai_controls_visibility(enable_ai):
    """根据AI开关状态更新控件可见性"""
    return gr.update(visible=enable_ai), gr.update(visible=enable_ai)

def update_mode_availability(enable_ai):
    """根据AI开关状态更新可用的处理模式"""
    processing_modes = hybrid_enhancer.get_available_modes()
    
    if enable_ai:
        # AI启用时，所有模式都可用
        choices = list(processing_modes.keys())
        value = "adaptive_hybrid"
    else:
        # AI禁用时，只有传统处理模式可用
        choices = ["traditional_only"]
        value = "traditional_only"
    
    return gr.update(choices=choices, value=value)

def create_hybrid_demo():
    """创建混合处理演示界面"""
    
    # 获取可用的处理模式和AI模型
    processing_modes = hybrid_enhancer.get_available_modes()
    ai_models = hybrid_enhancer.get_ai_models()
    
    with gr.Blocks(
        title="🎵 AI+传统混合音频增强系统",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; color: #2E86AB; font-size: 28px; font-weight: bold; margin: 20px 0; }
        .section-header { color: #A23B72; font-size: 18px; font-weight: bold; margin: 15px 0 10px 0; }
        .info-box { background: #F18F01; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .success-box { background: #C73E1D; color: white; padding: 15px; border-radius: 10px; margin: 10px 0; }
        .ai-controls { border: 2px solid #2E86AB; padding: 15px; border-radius: 10px; margin: 10px 0; }
        """
    ) as demo:
        
        gr.Markdown(
            """
            <div class="main-header">
            🎵 AI+传统混合音频增强系统 🎵
            </div>
            
            <div class="info-box">
            🌟 <strong>全新混合处理技术</strong><br>
            结合最新AI模型与经典信号处理算法，为您的音频提供最佳的增强效果！<br>
            🎯 <strong>完全自定义</strong>：您可以自由选择是否使用AI、使用哪个AI模型，以及如何组合不同的处理技术
            </div>
            """,
            elem_classes=["main-content"]
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown('<div class="section-header">📤 音频输入</div>')
                
                audio_input = gr.Audio(
                    label="上传音频文件",
                    type="filepath",
                    sources=["upload"]
                )
                
                gr.Markdown('<div class="section-header">⚙️ 处理设置</div>')
                
                # AI处理开关
                with gr.Group(elem_classes=["ai-controls"]):
                    gr.Markdown("🤖 **AI处理控制**")
                    enable_ai = gr.Checkbox(
                        label="启用AI增强处理",
                        value=True,
                        info="关闭此选项将仅使用传统信号处理算法"
                    )
                    
                    # AI相关控件（可以根据开关显示/隐藏）
                    with gr.Group() as ai_controls_group:
                        ai_model = gr.Dropdown(
                            choices=list(ai_models.keys()),
                            value="facebook_denoiser",
                            label="🤖 选择AI模型",
                            info="选择要使用的AI增强模型"
                        )
                        
                        load_model_btn = gr.Button("📥 预加载AI模型", variant="secondary", size="sm")
                
                processing_mode = gr.Dropdown(
                    choices=list(processing_modes.keys()),
                    value="adaptive_hybrid",
                    label="🔄 处理模式",
                    info="选择音频处理的策略"
                )
                
                with gr.Row():
                    enhancement_level = gr.Dropdown(
                        choices=["basic", "medium", "advanced"],
                        value="medium",
                        label="🔧 传统增强级别",
                        info="传统信号处理强度"
                    )
                    
                    blend_ratio = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="⚖️ 混合比例 (AI权重)",
                        info="仅在并行混合模式下生效"
                    )
                
                process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown('<div class="section-header">ℹ️ 模式说明</div>')
                
                mode_info = gr.Textbox(
                    value="选择处理模式查看详细说明",
                    label="处理模式说明",
                    lines=4,
                    interactive=False
                )
                
                gr.Markdown('<div class="section-header">🤖 AI模型信息</div>')
                
                model_info_display = gr.Textbox(
                    value="选择AI模型查看详细信息",
                    label="模型详情",
                    lines=6,
                    interactive=False
                )
                
                model_load_status = gr.Textbox(
                    label="模型加载状态",
                    lines=2,
                    interactive=False
                )
        
        gr.Markdown('<div class="section-header">📊 处理结果</div>')
        
        with gr.Row():
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="增强后的音频",
                    type="filepath"
                )
                
                processing_status = gr.Textbox(
                    label="处理状态",
                    lines=8,
                    interactive=False
                )
            
            with gr.Column(scale=1):
                process_details = gr.Textbox(
                    label="处理详情",
                    lines=12,
                    interactive=False
                )
        
        # 事件绑定
        def update_mode_info(mode):
            mode_descriptions = {
                "traditional_only": "🔧 仅使用传统信号处理技术，包括自适应降噪、谐波增强、动态范围优化等经典算法。稳定可靠，适合所有音频类型。",
                "ai_only": "🤖 仅使用AI深度学习模型进行音频增强。需要先加载对应的AI模型，处理效果取决于模型质量和音频特征。",
                "ai_then_traditional": "🤖➡️🔧 AI优先混合：先使用AI模型进行初步增强，再用传统算法进行精细调整。结合两者优势，适合多种音频类型。",
                "traditional_then_ai": "🔧➡️🤖 传统优先混合：先用传统算法进行基础处理，再用AI模型进行高级增强。适合噪声较多或质量较差的音频。",
                "parallel_blend": "🔀 并行混合：同时进行AI和传统处理，然后按设定比例混合结果。可以平衡不同方法的特点，效果最为灵活。",
                "adaptive_hybrid": "🧠 自适应混合：系统智能分析音频特征，自动选择最适合的处理策略。无需手动调整参数，适合各种音频质量。"
            }
            return mode_descriptions.get(mode, "请选择处理模式")
        
        # AI开关控制
        enable_ai.change(
            fn=lambda x: (gr.update(visible=x), gr.update(choices=list(processing_modes.keys()) if x else ["traditional_only"], value="adaptive_hybrid" if x else "traditional_only")),
            inputs=[enable_ai],
            outputs=[ai_controls_group, processing_mode]
        )
        
        processing_mode.change(
            fn=update_mode_info,
            inputs=[processing_mode],
            outputs=[mode_info]
        )
        
        ai_model.change(
            fn=get_model_info,
            inputs=[ai_model],
            outputs=[model_info_display]
        )
        
        process_btn.click(
            fn=process_audio_hybrid,
            inputs=[audio_input, processing_mode, enable_ai, ai_model, enhancement_level, blend_ratio],
            outputs=[audio_output, processing_status, process_details]
        )
        
        load_model_btn.click(
            fn=load_ai_model_interface,
            inputs=[ai_model],
            outputs=[model_load_status]
        )
        
        # 示例和帮助信息
        gr.Markdown(
            """
            ---
            ### 💡 使用建议
            
            **🎯 推荐设置组合：**
            
            | 音频类型 | AI处理 | 推荐模式 | AI模型 | 传统级别 | 混合比例 |
            |---------|--------|---------|--------|----------|----------|
            | 🎤 语音录音 | ✅ | 自适应混合 | Facebook Denoiser | Medium | - |
            | 🎵 音乐文件 | ✅ | 并行混合 | SpeechBrain | Medium | 0.6 |
            | 📻 低质量音频 | ✅ | AI优先混合 | RNNoise | Advanced | - |
            | 🎧 高质量音频 | 🔄 | 传统优先混合 | SpeechBrain | Basic | - |
            | 🎼 专业音频 | 🔄 | 并行混合 | Facebook Denoiser | Advanced | 0.3 |
            | 🗣️ 会议录音 | ✅ | 自适应混合 | RNNoise | Medium | - |
            
            **⚙️ 参数说明：**
            - **AI处理开关**: 完全控制是否使用AI模型，关闭后只使用传统算法
            - **AI模型选择**: 每个模型都有不同的特色和适用场景
            - **混合比例**: 1.0=完全AI, 0.0=完全传统, 0.5=平衡混合
            - **传统增强级别**: Basic(降噪) < Medium(+谐波) < Advanced(+动态+立体声)
            - **自适应模式**: 系统根据音频特征自动选择最佳策略
            
            **🤖 AI模型特点：**
            - **Facebook Denoiser**: 专业语音降噪，适合人声内容
            - **SpeechBrain**: 平衡的语音增强，适合多种音频类型
            - **RNNoise**: 轻量级实时降噪，资源占用少，速度快
            
            **🚀 性能提示：**
            - 首次使用AI模型需要下载和加载时间
            - 可以预先加载常用的AI模型以提高处理速度
            - GPU可用时AI处理速度更快
            - 传统处理速度快，CPU友好，质量稳定
            - 对于已经很高质量的音频，可以尝试关闭AI或降低AI权重
            """
        )
    
    return demo

if __name__ == "__main__":
    print("🎵 启动AI+传统混合音频增强系统...")
    
    demo = create_hybrid_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=False,
        show_error=True
    ) 