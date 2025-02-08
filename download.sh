#!/bin/bash

# Set default output directory if not provided
OUTPUT_DIR="${1:-data}"

# Base URL for all files
BASE_URL="https://huggingface.co"

# Create directory for downloads if it doesn't exist
mkdir -p "${OUTPUT_DIR}"

# Download each file
echo "Starting downloads to directory: ${OUTPUT_DIR}"

# .gitattributes
curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/.gitattributes?download=true" -o "${OUTPUT_DIR}/.gitattributes"

# GLOW graphs
curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/EtGlowExecutionProvider_GLOW_graph_Extracted_from_-Extracted_from_-Extracted_from_-Extracted_from_-main_graph----_2858289663424555064_0_0_0.onnx?download=true" -o "${OUTPUT_DIR}/glow_graph_2858289663424555064.onnx"
curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/EtGlowExecutionProvider_GLOW_graph_Extracted_from_-Extracted_from_-Extracted_from_-Extracted_from_-main_graph----_9203402667311998790_0_0_0.onnx?download=true" -o "${OUTPUT_DIR}/glow_graph_9203402667311998790.onnx"

# README
curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/README.md?download=true" -o "${OUTPUT_DIR}/README.md"

# Model files
curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/model-fixed-dims.onnx?download=true" -o "${OUTPUT_DIR}/model-fixed-dims.onnx"
curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/model.embed_tokens.weight?download=true" -o "${OUTPUT_DIR}/model.embed_tokens.weight"

# Layer weights
for i in {0..27}; do
    curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/model.layers.${i}.input_layernorm.weight?download=true" -o "${OUTPUT_DIR}/model.layers.${i}.input_layernorm.weight"
    curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/model.layers.${i}.post_attention_layernorm.weight?download=true" -o "${OUTPUT_DIR}/model.layers.${i}.post_attention_layernorm.weight"
    curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/model.layers.${i}.self_attn.q_proj.bias?download=true" -o "${OUTPUT_DIR}/model.layers.${i}.self_attn.q_proj.bias"
done

# Model norm weight
curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/model.norm.weight?download=true" -o "${OUTPUT_DIR}/model.norm.weight"

# Main model ONNX
curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/model.onnx?download=true" -o "${OUTPUT_DIR}/model.onnx"

# MatMul files
for i in {8851..8853} {8878..8884} {8909..8915} {8940..8946} {8971..8977} {9002..9008} {9033..9039} {9064..9070} {9095..9101} {9126..9132} {9157..9163} {9188..9194} {9219..9225} {9250..9256} {9281..9287} {9312..9318} {9343..9349} {9374..9380} {9405..9411} {9436..9442} {9467..9473} {9498..9504} {9529..9535} {9560..9566} {9591..9597} {9622..9628} {9653..9659} {9684..9690} {9715..9718} 9722; do
    curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/onnx__MatMul_${i}?download=true" -o "${OUTPUT_DIR}/onnx__MatMul_${i}"
done

# Tokenizer files
curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/special_tokens_map.json?download=true" -o "${OUTPUT_DIR}/special_tokens_map.json"
curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/tokenizer.json?download=true" -o "${OUTPUT_DIR}/tokenizer.json"
curl -L "${BASE_URL}/Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main/tokenizer_config.json?download=true" -o "${OUTPUT_DIR}/tokenizer_config.json"

echo "Download script completed! Files downloaded to: ${OUTPUT_DIR}"