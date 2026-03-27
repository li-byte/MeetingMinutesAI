import os
import shlex
import subprocess
from functools import wraps
import warnings
import gc
from typing import Optional

import torch
import whisper
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Segment
from whisper import Whisper

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain.utils.torch_audio_backend")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 全局变量
_pipeline: Optional[Pipeline] = None
_whisper_model: Optional[Whisper] = None
_whisper_device: Optional[str] = None

def timer():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


def initialize_models(model="small", config_path=None):
    """
    预加载所有模型，应该在应用启动时调用
    """
    global _pipeline, _whisper_model, _whisper_device

    print("正在加载模型...")

    # 设置设备
    _whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {_whisper_device}")

    # 加载 Whisper 模型
    print(f"加载 Whisper 模型 ({model})...")
    _whisper_model = whisper.load_model(model, device=_whisper_device,download_root="model/whisper")

    # 加载 Pyannote 模型
    if config_path is None:
        # 如果没有指定配置路径，使用默认路径
        config_path = "model/speaker-diarization-3-1/config.yaml"
    print(f"加载 Pyannote 模型...")
    _pipeline = Pipeline.from_pretrained(config_path)
    if torch.cuda.is_available():
        _pipeline.to(torch.device("cuda"))

    print("模型加载完成！")


def get_pipeline()->Pipeline:
    """获取 Pyannote pipeline 实例"""
    global _pipeline
    if _pipeline is None:
        raise RuntimeError("模型未初始化，请先调用 initialize_models()")
    return _pipeline


def get_whisper_model()-> tuple[Whisper, str]:
    """获取 Whisper 模型实例"""
    global _whisper_model, _whisper_device
    if _whisper_model is None:
        raise RuntimeError("模型未初始化，请先调用 initialize_models()")
    return _whisper_model, _whisper_device


def cleanup_gpu():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def is_audio_file(file_path):
    """判断文件是否为音频文件"""
    audio_exts = ['.mp3', '.aac', '.wav', '.m4a', '.flac', '.ogg']
    video_exts = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    ext = os.path.splitext(file_path)[1].lower()
    if ext in audio_exts:
        return True
    elif ext in video_exts:
        return False
    else:
        raise ValueError(f"无法识别的文件类型: {ext}")


@timer()
def extract_audio(video_path, output_path):
    """从视频中提取音频"""
    probe_cmd = f'ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
    try:
        result = subprocess.run(shlex.split(probe_cmd), capture_output=True, text=True, check=True)
        input_codec = result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

    if input_codec.lower() != 'mp3':
        cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'mp3', '-ar', '44100', '-ac', '2', output_path]
    else:
        cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'copy', output_path]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except subprocess.CalledProcessError:
        return None


@timer()
def ensure_16khz(audio_path):
    """确保音频为16kHz采样率"""
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        base, ext = os.path.splitext(audio_path)
        fixed_path = f"{base}_16khz.wav"
        torchaudio.save(fixed_path, waveform, 16000)
        return fixed_path
    return audio_path


def get_speaker_for_time(start_time, end_time, diarization):
    """获取指定时间段的说话人"""
    segment_to_check = Segment(start_time, end_time)
    speaker_overlap_duration = {}

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        if segment_to_check.intersects(segment):
            overlap_start = max(segment_to_check.start, segment.start)
            overlap_end = min(segment_to_check.end, segment.end)
            overlap_duration = overlap_end - overlap_start
            if speaker in speaker_overlap_duration:
                speaker_overlap_duration[speaker] += overlap_duration
            else:
                speaker_overlap_duration[speaker] = overlap_duration

    if not speaker_overlap_duration:
        return "Unknown"
    return max(speaker_overlap_duration, key=speaker_overlap_duration.get)


@timer()
def process_media(file_path: str) -> list:
    """
    处理媒体文件，返回带说话人标签的转录结果

    注意：whisper_model 参数在这里被保留但不再使用，因为模型已预加载
    """
    temp_files = []

    try:
        # 获取预加载的模型
        pipeline = get_pipeline()
        model, device = get_whisper_model()

        # 检查文件类型并提取音频
        if is_audio_file(file_path):
            audio_path = file_path
        else:
            base, _ = os.path.splitext(file_path)
            audio_path = f"{base}_extracted.mp3"
            extracted = extract_audio(file_path, audio_path)
            if not extracted:
                raise RuntimeError("音频提取失败")
            temp_files.append(audio_path)

        # 确保16kHz采样率
        audio_16khz = ensure_16khz(audio_path)
        if audio_16khz != audio_path:
            temp_files.append(audio_16khz)

        # 说话人分离
        diarization = pipeline(audio_16khz).speaker_diarization

        # Whisper 转录
        result = model.transcribe(audio_16khz, language="zh", fp16=device == "cuda")

        # 构建结果
        json_segments = []
        for seg in result['segments']:
            json_segments.append({
                "start": round(seg['start'], 2),
                "end": round(seg['end'], 2),
                "speaker": get_speaker_for_time(seg['start'], seg['end'], diarization),
                "text": seg['text'].strip()
            })

        return json_segments

    finally:
        # 清理临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

# 可选：在模块导入时自动初始化
# 如果需要自动初始化，取消下面的注释
initialize_models(model="small")