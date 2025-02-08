# raijin
lightning fast CPU inference in ONNX for deepseek-r1-distill-qwen-1.5b

## setup

download the model & tokenizer config into `data/`

```bash
$ mkdir -p data
$ curl -Lo data/model_quantized.onnx https://huggingface.co/onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX/resolve/main/onnx/model_quantized.onnx
$ curl -Lo data/tokenizer.json https://huggingface.co/onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX/resolve/main/tokenizer.json
```
