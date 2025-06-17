import numpy as np
import librosa
from typing import Tuple, Optional, List
from ai_models import ai_enhancer
from app import AudioQualityEnhancer
import warnings
warnings.filterwarnings("ignore")

class HybridAudioEnhancer:
    """混合音频增强器 - 结合传统信号处理和AI模型"""
    
    def __init__(self):
        # 传统信号处理器
        self.traditional_enhancer = AudioQualityEnhancer()
        
        # AI增强器
        self.ai_enhancer = ai_enhancer
        
        # 处理模式
        self.processing_modes = {
            "traditional_only": "仅传统处理",
            "ai_only": "仅AI处理", 
            "ai_then_traditional": "AI优先混合",
            "traditional_then_ai": "传统优先混合",
            "parallel_blend": "并行混合",
            "adaptive_hybrid": "自适应混合"
        }
        
        print("🎵 混合音频增强器初始化完成")
    
    def get_available_modes(self) -> dict:
        """获取可用的处理模式"""
        return self.processing_modes
    
    def get_ai_models(self) -> dict:
        """获取可用的AI模型"""
        return self.ai_enhancer.get_available_models()
    
    def load_ai_model(self, model_name: str) -> bool:
        """加载指定的AI模型"""
        return self.ai_enhancer.download_and_load_model(model_name)
    
    def process_traditional_only(self, audio: np.ndarray, sr: int, 
                               enhancement_level: str = "medium") -> np.ndarray:
        """仅使用传统方法处理"""
        print("🔧 使用传统信号处理...")
        
        enhanced = audio.copy()
        
        if enhancement_level in ["basic", "medium", "advanced"]:
            # 降噪
            enhanced = self.traditional_enhancer.adaptive_noise_reduction(enhanced, sr)
            
        if enhancement_level in ["medium", "advanced"]:
            # 谐波增强
            enhanced = self.traditional_enhancer.harmonic_enhancement(enhanced, sr)
            
        if enhancement_level == "advanced":
            # 动态范围处理
            enhanced = self.traditional_enhancer.dynamic_range_enhancement(enhanced)
            
            # 立体声宽度增强（如果是立体声）
            if len(enhanced.shape) > 1 or enhanced.ndim > 1:
                enhanced = self.traditional_enhancer.stereo_width_enhancement(enhanced)
        
        return enhanced
    
    def process_ai_only(self, audio: np.ndarray, sr: int, 
                       ai_model: str = "facebook_denoiser") -> np.ndarray:
        """仅使用AI模型处理"""
        print(f"🤖 使用AI模型处理: {ai_model}")
        
        if not self.ai_enhancer.is_model_loaded(ai_model):
            print(f"⏳ 正在加载AI模型: {ai_model}")
            if not self.ai_enhancer.download_and_load_model(ai_model):
                print(f"❌ AI模型加载失败，使用传统处理作为备选")
                return self.process_traditional_only(audio, sr, "medium")
        
        enhanced = self.ai_enhancer.enhance_audio(audio, sr, ai_model)
        return enhanced
    
    def process_ai_then_traditional(self, audio: np.ndarray, sr: int,
                                  ai_model: str = "facebook_denoiser",
                                  enhancement_level: str = "basic") -> np.ndarray:
        """AI优先混合：先AI处理，再传统增强"""
        print(f"🤖➡️🔧 AI优先混合处理: {ai_model} + 传统{enhancement_level}")
        
        # 第一步：AI处理
        ai_enhanced = self.process_ai_only(audio, sr, ai_model)
        
        # 第二步：传统增强（使用较轻的级别避免过度处理）
        final_enhanced = self.process_traditional_only(ai_enhanced, sr, enhancement_level)
        
        return final_enhanced
    
    def process_traditional_then_ai(self, audio: np.ndarray, sr: int,
                                  ai_model: str = "facebook_denoiser", 
                                  enhancement_level: str = "basic") -> np.ndarray:
        """传统优先混合：先传统处理，再AI增强"""
        print(f"🔧➡️🤖 传统优先混合处理: 传统{enhancement_level} + {ai_model}")
        
        # 第一步：传统处理
        traditional_enhanced = self.process_traditional_only(audio, sr, enhancement_level)
        
        # 第二步：AI增强
        final_enhanced = self.process_ai_only(traditional_enhanced, sr, ai_model)
        
        return final_enhanced
    
    def process_parallel_blend(self, audio: np.ndarray, sr: int,
                             ai_model: str = "facebook_denoiser",
                             enhancement_level: str = "medium",
                             blend_ratio: float = 0.5) -> np.ndarray:
        """并行混合：同时进行AI和传统处理，然后混合结果"""
        print(f"🔀 并行混合处理: {ai_model} + 传统{enhancement_level} (混合比例: {blend_ratio:.1f})")
        
        # 并行处理
        ai_enhanced = self.process_ai_only(audio, sr, ai_model)
        traditional_enhanced = self.process_traditional_only(audio, sr, enhancement_level)
        
        # 确保两个结果长度一致
        min_len = min(len(ai_enhanced), len(traditional_enhanced))
        ai_enhanced = ai_enhanced[:min_len]
        traditional_enhanced = traditional_enhanced[:min_len]
        
        # 混合结果
        blended = blend_ratio * ai_enhanced + (1 - blend_ratio) * traditional_enhanced
        
        return blended
    
    def process_adaptive_hybrid(self, audio: np.ndarray, sr: int,
                              ai_model: str = "facebook_denoiser") -> np.ndarray:
        """自适应混合：根据音频特征智能选择最佳处理方式"""
        print("🧠 自适应混合处理...")
        
        # 分析音频特征
        audio_features = self._analyze_audio_features(audio, sr)
        
        # 根据特征选择处理策略
        if audio_features["noise_level"] > 0.3:
            print("📊 检测到高噪声，优先使用AI降噪")
            return self.process_ai_then_traditional(audio, sr, ai_model, "basic")
        elif audio_features["dynamic_range"] < 0.2:
            print("📊 检测到动态范围窄，优先使用传统增强")
            return self.process_traditional_then_ai(audio, sr, ai_model, "advanced")
        elif audio_features["spectral_centroid"] > 3000:
            print("📊 检测到高频内容丰富，使用并行混合")
            return self.process_parallel_blend(audio, sr, ai_model, "medium", 0.6)
        else:
            print("📊 使用平衡的混合处理")
            return self.process_parallel_blend(audio, sr, ai_model, "medium", 0.5)
    
    def _analyze_audio_features(self, audio: np.ndarray, sr: int) -> dict:
        """分析音频特征"""
        try:
            # 噪声水平估计
            stft = librosa.stft(audio)
            spectral_rolloff = librosa.feature.spectral_rolloff(S=np.abs(stft), sr=sr)
            noise_level = np.std(spectral_rolloff) / np.mean(spectral_rolloff)
            
            # 动态范围
            rms = librosa.feature.rms(y=audio)
            dynamic_range = np.std(rms) / np.mean(rms)
            
            # 频谱质心
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            
            # 零交叉率（语音活动检测）
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            
            return {
                "noise_level": min(noise_level, 1.0),
                "dynamic_range": min(dynamic_range, 1.0), 
                "spectral_centroid": spectral_centroid,
                "zero_crossing_rate": zcr
            }
        except Exception as e:
            print(f"音频特征分析失败: {str(e)}")
            # 返回中性特征
            return {
                "noise_level": 0.2,
                "dynamic_range": 0.3,
                "spectral_centroid": 2000,
                "zero_crossing_rate": 0.1
            }
    
    def _get_blend_ratio_display(self, processing_mode: str, actual_ai_used: bool, actual_traditional_used: bool, blend_ratio: float) -> str:
        """获取混合比例的显示文本"""
        if processing_mode == "parallel_blend":
            return f"{blend_ratio:.1f} (AI权重)"
        elif processing_mode == "adaptive_hybrid":
            if actual_ai_used and actual_traditional_used:
                return "自适应混合 (系统自动决定)"
            elif actual_ai_used:
                return "1.0 (纯AI)"
            else:
                return "0.0 (纯传统)"
        elif processing_mode == "ai_then_traditional":
            return "AI→传统 (顺序处理)"
        elif processing_mode == "traditional_then_ai":
            return "传统→AI (顺序处理)"
        elif processing_mode == "ai_only":
            return "1.0 (纯AI)"
        elif processing_mode == "traditional_only":
            return "0.0 (纯传统)"
        else:
            return "不适用"
    
    def enhance_audio(self, audio: np.ndarray, sr: int, 
                     processing_mode: str = "adaptive_hybrid",
                     ai_model: str = "facebook_denoiser",
                     enhancement_level: str = "medium",
                     blend_ratio: float = 0.5) -> Tuple[np.ndarray, dict]:
        """主要的音频增强接口"""
        
        # 输入验证
        if len(audio) == 0:
            print("❌ 输入音频为空")
            return audio, {"error": "空音频"}
        
        if not np.isfinite(audio).all():
            print("❌ 输入音频包含无效数值")
            return audio, {"error": "无效音频数据"}
        
        # 音频特征分析
        features = self._analyze_audio_features(audio, sr)
        
        try:
            # 记录实际使用的处理方法
            actual_ai_used = False
            actual_traditional_used = False
            actual_method_details = ""
            
            # 根据模式选择处理方法
            if processing_mode == "traditional_only":
                enhanced = self.process_traditional_only(audio, sr, enhancement_level)
                method_used = f"传统处理 ({enhancement_level})"
                actual_traditional_used = True
                actual_method_details = f"仅传统信号处理，级别：{enhancement_level}"
                
            elif processing_mode == "ai_only":
                enhanced = self.process_ai_only(audio, sr, ai_model)
                method_used = f"AI处理 ({ai_model})"
                actual_ai_used = True
                actual_method_details = f"仅AI模型处理：{ai_model}"
                
            elif processing_mode == "ai_then_traditional":
                enhanced = self.process_ai_then_traditional(audio, sr, ai_model, enhancement_level)
                method_used = f"AI→传统 ({ai_model} + {enhancement_level})"
                actual_ai_used = True
                actual_traditional_used = True
                actual_method_details = f"AI优先：{ai_model} → 传统{enhancement_level}"
                
            elif processing_mode == "traditional_then_ai":
                enhanced = self.process_traditional_then_ai(audio, sr, ai_model, enhancement_level)
                method_used = f"传统→AI ({enhancement_level} + {ai_model})"
                actual_ai_used = True
                actual_traditional_used = True
                actual_method_details = f"传统优先：{enhancement_level} → {ai_model}"
                
            elif processing_mode == "parallel_blend":
                enhanced = self.process_parallel_blend(audio, sr, ai_model, enhancement_level, blend_ratio)
                method_used = f"并行混合 ({ai_model} + {enhancement_level}, 比例:{blend_ratio:.1f})"
                actual_ai_used = True
                actual_traditional_used = True
                actual_method_details = f"并行混合：{ai_model}({blend_ratio:.1f}) + 传统{enhancement_level}({1-blend_ratio:.1f})"
                
            elif processing_mode == "adaptive_hybrid":
                enhanced = self.process_adaptive_hybrid(audio, sr, ai_model)
                method_used = "自适应混合"
                # 自适应模式会根据音频特征决定是否使用AI
                if features["noise_level"] > 0.3 or features["dynamic_range"] < 0.2 or features["spectral_centroid"] > 3000:
                    actual_ai_used = True
                    actual_traditional_used = True
                    actual_method_details = f"自适应选择：{ai_model} + 传统处理"
                else:
                    actual_ai_used = True
                    actual_traditional_used = True
                    actual_method_details = f"自适应选择：平衡混合 {ai_model} + 传统处理"
                
            else:
                print(f"❌ 未知的处理模式: {processing_mode}")
                enhanced = self.process_traditional_only(audio, sr, "medium")
                method_used = "传统处理 (默认)"
                actual_traditional_used = True
                actual_method_details = "默认传统处理"
            
            # 最终检查和标准化
            if not np.isfinite(enhanced).all():
                print("⚠️ 处理结果包含无效数值，使用原始音频")
                enhanced = audio
                method_used += " (失败，返回原始)"
            
            # 标准化到合理范围
            if np.max(np.abs(enhanced)) > 0:
                enhanced = enhanced / np.max(np.abs(enhanced)) * 0.95
            
            # 返回结果和元数据
            metadata = {
                "method_used": method_used,
                "original_features": features,
                "processing_mode": processing_mode,
                "ai_model": ai_model if actual_ai_used else None,
                "ai_model_used": ai_model if actual_ai_used else "未使用",
                "traditional_used": actual_traditional_used,
                "enhancement_level": enhancement_level,
                "actual_method_details": actual_method_details,
                "blend_ratio_used": blend_ratio if processing_mode == "parallel_blend" else None,
                "blend_ratio_display": self._get_blend_ratio_display(processing_mode, actual_ai_used, actual_traditional_used, blend_ratio),
                "success": True
            }
            
            print(f"✅ 音频增强完成: {method_used}")
            return enhanced, metadata
            
        except Exception as e:
            print(f"❌ 音频增强失败: {str(e)}")
            return audio, {"error": str(e), "success": False}

# 全局混合增强器实例
hybrid_enhancer = HybridAudioEnhancer() 