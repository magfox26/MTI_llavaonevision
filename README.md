# MTI_llavaonevision

### 1.环境安装
使用pip进行安装：
```bash
pip install 'ms-swift'
pip install av deepspeed
```
### 2.下载模型
模型用此[链接](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-si-hf)，用lmms-lab在ms-swift框架下会报错  
```bash
huggingface-cli download llava-hf/llava-onevision-qwen2-7b-ov-hf --local-dir ./llava-hf/llava-onevision-qwen2-7b-ov-hf
```
local-dir：指定下载模型保存到本地的目录
### 3.训练
在[inference_code](https://github.com/magfox26/MTI_llavaonevision/tree/main/inference_code)文件夹下的‘mit10m_train.sh’
更改‘mit10m_train.sh’中的设置：



