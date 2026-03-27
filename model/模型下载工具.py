import os
# 设置使用 Hugging Face 镜像站，加速国内下载
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

if __name__ == '__main__':

    snapshot_download(
        repo_id="pyannote/segmentation-3.0",
        local_dir="/model/pyannote/wespeaker-voxceleb-resnet34-LM",
        token="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"
    )