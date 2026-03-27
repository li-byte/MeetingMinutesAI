# import os
# import shlex
# import subprocess
# import time
# from functools import wraps
#
# import torch
# import whisper
# import torchaudio
# from pyannote.audio import Pipeline
# from pyannote.core import Segment
# import warnings
# import gc
#
# warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")
# warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain.utils.torch_audio_backend")
# warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio.models.blocks.pooling")
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
#
# # 全局模型变量，实现模型复用
# _pipeline = None
# _whisper_model = None
# _whisper_device = None
#
#
# # 添加计时装饰器
# def timer(func_name=None):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             start = time.time()
#             result = func(*args, **kwargs)
#             elapsed = time.time() - start
#             name = func_name or func.__name__
#             print(f"⏱️  {name} 耗时: {elapsed:.2f} 秒")
#             return result
#
#         return wrapper
#
#     return decorator
#
#
# def get_pipeline():
#     """获取或加载说话人识别模型（单例模式）"""
#     global _pipeline
#     if _pipeline is None:
#         print("\n🎤 首次加载说话人识别模型...")
#         load_start = time.time()
#         _pipeline = Pipeline.from_pretrained(
#             "E:/pycharm_python_project/PythonProject3/model/config.yaml"
#         )
#         if torch.cuda.is_available():
#             _pipeline.to(torch.device("cuda"))
#             print(f"  模型已加载到 GPU: CUDA")
#         else:
#             print(f"  模型已加载到 CPU")
#         print(f"  说话人模型加载耗时: {time.time() - load_start:.2f} 秒")
#     else:
#         print("\n🎤 使用已加载的说话人识别模型")
#     return _pipeline
#
#
# def get_whisper_model(model_size="small"):
#     """获取或加载Whisper模型（单例模式）"""
#     global _whisper_model, _whisper_device
#
#     if _whisper_model is None:
#         print("\n📝 首次加载Whisper模型...")
#         load_start = time.time()
#         _whisper_device = "cuda" if torch.cuda.is_available() else "cpu"
#         _whisper_model = whisper.load_model(model_size, device=_whisper_device)
#         print(f"  Whisper模型加载耗时: {time.time() - load_start:.2f} 秒")
#         print(f"  模型: {model_size}, 设备: {_whisper_device}")
#     else:
#         print("\n📝 使用已加载的Whisper模型")
#
#     return _whisper_model, _whisper_device
#
#
# def cleanup_gpu():
#     """清理GPU缓存"""
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         gc.collect()
#         print("  GPU缓存已清理")
#
#
# def is_audio_file(file_path):
#     audio_exts = ['.mp3', '.aac', '.wav', '.m4a', '.flac', '.ogg']
#     video_exts = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
#     ext = os.path.splitext(file_path)[1].lower()
#     if ext in audio_exts:
#         return True
#     elif ext in video_exts:
#         return False
#     else:
#         raise ValueError(f"无法识别的文件类型: {ext}")
#
#
# @timer("音频提取")
# def extract_audio(video_path, output_path):
#     probe_cmd = f'ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "{video_path}"'
#     try:
#         result = subprocess.run(shlex.split(probe_cmd), capture_output=True, text=True, check=True)
#         input_codec = result.stdout.strip()
#     except subprocess.CalledProcessError:
#         return None
#
#     if input_codec.lower() != 'mp3':
#         cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'mp3', '-ar', '44100', '-ac', '2', output_path]
#     else:
#         cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'copy', output_path]
#
#     try:
#         subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         return output_path
#     except subprocess.CalledProcessError:
#         return None
#
#
# @timer("重采样到16kHz")
# def ensure_16khz(audio_path):
#     waveform, sr = torchaudio.load(audio_path)
#     print(f"  原始采样率: {sr} Hz")
#     if sr != 16000:
#         waveform = torchaudio.functional.resample(waveform, sr, 16000)
#         base, ext = os.path.splitext(audio_path)
#         fixed_path = f"{base}_16khz.wav"
#         torchaudio.save(fixed_path, waveform, 16000)
#         print(f"  重采样完成，新采样率: 16000 Hz")
#         return fixed_path
#     print(f"  已是16kHz，无需重采样")
#     return audio_path
#
#
# def get_speaker_for_time(start_time, end_time, diarization):
#     segment_to_check = Segment(start_time, end_time)
#     speaker_overlap_duration = {}
#
#     for segment, _, speaker in diarization.itertracks(yield_label=True):
#         if segment_to_check.intersects(segment):
#             overlap_start = max(segment_to_check.start, segment.start)
#             overlap_end = min(segment_to_check.end, segment.end)
#             overlap_duration = overlap_end - overlap_start
#
#             if speaker in speaker_overlap_duration:
#                 speaker_overlap_duration[speaker] += overlap_duration
#             else:
#                 speaker_overlap_duration[speaker] = overlap_duration
#
#     if not speaker_overlap_duration:
#         return "Unknown"
#
#     return max(speaker_overlap_duration, key=speaker_overlap_duration.get)
#
#
# @timer("process_media总耗时")
# def process_media(file_path: str, whisper_model_size="small") -> list:
#     """
#     处理媒体文件，返回带说话人识别的转写结果
#
#     Args:
#         file_path: 媒体文件路径（支持音频和视频）
#         whisper_model_size: Whisper模型大小 ("tiny", "base", "small", "medium", "large")
#
#     Returns:
#         JSON格式的转写结果列表
#     """
#     temp_files = []  # 记录临时文件，用于清理
#
#     try:
#         # 1. 音频提取/识别阶段
#         print("\n📁 阶段1: 音频准备")
#         stage_start = time.time()
#
#         if is_audio_file(file_path):
#             audio_path = file_path
#             print("  文件类型: 音频文件")
#         else:
#             base, _ = os.path.splitext(file_path)
#             audio_path = f"{base}_extracted.mp3"
#             print("  文件类型: 视频文件，开始提取音频...")
#             extracted = extract_audio(file_path, audio_path)
#             if not extracted:
#                 raise RuntimeError("音频提取失败")
#             temp_files.append(audio_path)
#
#         audio_16khz = ensure_16khz(audio_path)
#         if audio_16khz != audio_path:
#             temp_files.append(audio_16khz)
#
#         print(f"  音频准备耗时: {time.time() - stage_start:.2f} 秒")
#
#         # 2. 说话人识别阶段（使用全局模型）
#         print("\n🎤 阶段2: 说话人识别")
#         stage_start = time.time()
#
#         pipeline = get_pipeline()
#         diar_start = time.time()
#         diarization = pipeline(audio_16khz).speaker_diarization
#         print(f"  说话人识别推理耗时: {time.time() - diar_start:.2f} 秒")
#         print(f"  说话人识别总耗时: {time.time() - stage_start:.2f} 秒")
#
#         # 3. Whisper转写阶段（使用全局模型）
#         print("\n📝 阶段3: Whisper转写")
#         stage_start = time.time()
#
#         modelclient, device = get_whisper_model(whisper_model_size)
#         transcribe_start = time.time()
#         result = modelclient.transcribe(audio_16khz, language="zh", fp16=device == "cuda")
#         print(f"  Whisper转写耗时: {time.time() - transcribe_start:.2f} 秒")
#         print(f"  Whisper总耗时: {time.time() - stage_start:.2f} 秒")
#
#         # 4. 合并结果阶段
#         print("\n🔗 阶段4: 合并说话人识别和转写结果")
#         stage_start = time.time()
#
#         json_segments = []
#         for seg in result['segments']:
#             json_segments.append({
#                 "start": round(seg['start'], 2),
#                 "end": round(seg['end'], 2),
#                 "speaker": get_speaker_for_time(seg['start'], seg['end'], diarization),
#                 "text": seg['text'].strip()
#             })
#
#         print(f"  合并结果耗时: {time.time() - stage_start:.2f} 秒")
#         print(f"  共生成 {len(json_segments)} 个段落")
#
#         return json_segments
#
#     finally:
#         # 清理临时文件
#         for temp_file in temp_files:
#             if os.path.exists(temp_file):
#                 os.remove(temp_file)
#                 print(f"  清理临时文件: {temp_file}")
#
#         # 可选：清理GPU缓存（在长时间运行后调用）
#         # cleanup_gpu()
#
#
# if __name__ == '__main__':
#     print("=" * 50)
#     print("开始处理...")
#     print("=" * 50)
#     total_start = time.time()
#
#     # 第一次处理 - 会加载模型
#     print("\n【第一次处理】")
#     result1 = process_media("E:\\pycharm_python_project\\PythonProject3\\x.mp3")
#
#     print("\n" + "=" * 50)
#     print(f"✅ 第一次处理完成！总耗时: {time.time() - total_start:.2f} 秒")
#     print("=" * 50)
#
#     # 输出前几个结果作为示例
#     print("\n📄 转写结果示例（前3段）:")
#     for i, seg in enumerate(result1[:3]):
#         print(f"  [{seg['start']:.2f} - {seg['end']:.2f}] {seg['speaker']}: {seg['text']}")
#
#     # 测试第二次处理 - 模型复用，速度会快很多
#     print("\n" + "=" * 50)
#     print("【第二次处理 - 测试模型复用】")
#     print("=" * 50)
#
#     second_start = time.time()
#     result2 = process_media("E:\\pycharm_python_project\\PythonProject3\\x.mp3")
#     print(f"\n✅ 第二次处理完成！总耗时: {time.time() - second_start:.2f} 秒")
#
#     print("\n" + "=" * 50)
#     print(f"🎉 全部完成！总耗时: {time.time() - total_start:.2f} 秒")
#     print("=" * 50)