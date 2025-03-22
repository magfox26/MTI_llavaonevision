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
huggingface-cli download llava-hf/llava-onevision-qwen2-7b-si-hf --local-dir ./llava-hf/llava-onevision-qwen2-7b-si-hf
```
`local-dir`指定下载模型保存到本地的目录
### 3.训练
1.更改[train](https://github.com/magfox26/MTI_llavaonevision/tree/main/train)文件夹下的‘mit10m_train.sh’ 中的设置：  
- `NPROC_PER_NODE` 使用多少张卡
- `CUDA_VISIBLE_DEVICES` 指定卡号
- `--model`下载好的模型路径
- `dataset`和`val_dataset`训练集和验证集路径
- `--deepspeed`zero2路径，在`./swift/swift/llm/ds_config`下
- `--output_dir`指定输出路径及文件夹名称

2.开始训练 `bash mit10m_train.sh`(train_loss等图片默认保存在`--output_dir`下的images文件夹下)  

3.训练完成后进行lora权重合并，更改[train](https://github.com/magfox26/MTI_llavaonevision/tree/main/train)文件夹下的`merge_lora.sh`中的设置：  

- `CUDA_VISIBLE_DEVICES` 指定卡号
- `--ckpt_dir`checkpoint-xx保存的位置
- `--output_dir`指定输出路径及文件夹名称（没指定的话默认保存在checkpoint同一目录下名为`checkpoint-xx-merged`）
  
4.开始合并 `bash merge_lora.sh`

**开始训练时如果遇到报错：**<font color="red">`TypeError: type Tensor doesn't define __round__ method</font>`  

找到报错的地方`./swift_env/lib/python3.10/site-packages/transformers/models/llava_onevision/processing_llava_onevision.py`并修改：  
```bash
new_width = int(round(float(width) * (float(current_height) / float(height)), 7))
```
```bash
new_height = int(round(float(height) * (float(current_width) / float(width)), 7))
```
### 4.推理
1.更改[inference](https://github.com/magfox26/MTI_llavaonevision/tree/main/inference)文件夹下的‘inference.py’ 中的设置：   
- `model_id` 指定模型路径
- `root` 数据存放的根目录
- `output_path` 输出的根目录
- `output_name` 输出文件名称
  
2.开始推理`CUDA_VISIBLE_DEVICES=0 python inference.py`





