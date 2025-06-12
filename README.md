# Rockchip NPU Multimodal Conversion 

### This repository contains a full pipeline to convert Qwen2VL safetensors models to run on the RK3588 NPU

Currently it support only
```bash
[?] Which Qwen2VL model would you like to convert?:
   Qwen/Qwen2-VL-7B-Instruct
 > Qwen/Qwen2-VL-2B-Instruct
```

Requirements:
```
rkllm-toolkit==1.2.1b1
rknn-toolkit2>=2.3.2
python==3.8
```

## Modification from the orginal source

## Instruction to use

1. Clone this repo and cd to the directory
3. You can choose either CPU or GPU to convert. If you use CPU, make sure you have enough RAM. If you don't, use `fallocate` to expand RAM with a swapfile.
4. Build and run the Docker container / Or run natively `python export_vision.py --step 0 --batch 1 --height 392 --width 392`
5. Select your model (2B or 7B)
6. Enjoy

## Reference
- https://github.com/airockchip/rknn-llm/blob/main/examples/Qwen2-VL_Demo/
- https://github.com/c0zaut/rkllm-mm-export
