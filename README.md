# MTI_llavaonevision

## 1.环境安装
使用pip进行安装：
```bash
pip install 'ms-swift'
pip install av deepspeed

## 2.下载模型
模型务必用此[链接](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-si-hf)，用lmms-lab在ms-swift框架下会报错  

local-dir：指定下载模型保存到本地的目录
```bash
huggingface-cli download llava-hf/llava-onevision-qwen2-7b-ov-hf --local-dir ./llava-hf/llava-onevision-qwen2-7b-ov-hf


