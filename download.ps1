param(
    [string]$OutputDir = "data"  # Default value if no parameter is provided
)

# Base URL for all files
$BaseUrl = "https://huggingface.co"
$RepoPath = "Esperanto/DeepSeek-R1-Distill-Qwen-1.5B-kvc-fp16-onnx/resolve/main"

# Create directory for downloads if it doesn't exist
Write-Host "Creating output directory: $OutputDir"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# Function to download a file with progress
function Download-FileWithProgress {
    param (
        [string]$Url,
        [string]$OutputPath
    )
    
    $fileName = Split-Path $OutputPath -Leaf
    Write-Host "Downloading: $fileName"
    
    try {
        Invoke-WebRequest -Uri $Url -OutFile $OutputPath -Headers @{"Cache-Control"="no-cache"}
        Write-Host "Successfully downloaded: $fileName" -ForegroundColor Green
    }
    catch {
        Write-Host "Failed to download: $fileName" -ForegroundColor Red
        Write-Host "Error: $_" -ForegroundColor Red
    }
}

Write-Host "Starting downloads to directory: $OutputDir" -ForegroundColor Cyan
$startTime = Get-Date

# Download .gitattributes
Download-FileWithProgress -Url "$BaseUrl/$RepoPath/.gitattributes?download=true" -OutputPath "$OutputDir\.gitattributes"

# Download GLOW graphs
Download-FileWithProgress -Url "$BaseUrl/$RepoPath/EtGlowExecutionProvider_GLOW_graph_Extracted_from_-Extracted_from_-Extracted_from_-Extracted_from_-main_graph----_2858289663424555064_0_0_0.onnx?download=true" -OutputPath "$OutputDir\glow_graph_2858289663424555064.onnx"
Download-FileWithProgress -Url "$BaseUrl/$RepoPath/EtGlowExecutionProvider_GLOW_graph_Extracted_from_-Extracted_from_-Extracted_from_-Extracted_from_-main_graph----_9203402667311998790_0_0_0.onnx?download=true" -OutputPath "$OutputDir\glow_graph_9203402667311998790.onnx"

# Download README
Download-FileWithProgress -Url "$BaseUrl/$RepoPath/README.md?download=true" -OutputPath "$OutputDir\README.md"

# Download model files
Download-FileWithProgress -Url "$BaseUrl/$RepoPath/model-fixed-dims.onnx?download=true" -OutputPath "$OutputDir\model-fixed-dims.onnx"
Download-FileWithProgress -Url "$BaseUrl/$RepoPath/model.embed_tokens.weight?download=true" -OutputPath "$OutputDir\model.embed_tokens.weight"

# Download layer weights
0..27 | ForEach-Object {
    $i = $_
    Download-FileWithProgress -Url "$BaseUrl/$RepoPath/model.layers.$i.input_layernorm.weight?download=true" -OutputPath "$OutputDir\model.layers.$i.input_layernorm.weight"
    Download-FileWithProgress -Url "$BaseUrl/$RepoPath/model.layers.$i.post_attention_layernorm.weight?download=true" -OutputPath "$OutputDir\model.layers.$i.post_attention_layernorm.weight"
    Download-FileWithProgress -Url "$BaseUrl/$RepoPath/model.layers.$i.self_attn.q_proj.bias?download=true" -OutputPath "$OutputDir\model.layers.$i.self_attn.q_proj.bias"
}

# Download model norm weight
Download-FileWithProgress -Url "$BaseUrl/$RepoPath/model.norm.weight?download=true" -OutputPath "$OutputDir\model.norm.weight"

# Download main model ONNX
Download-FileWithProgress -Url "$BaseUrl/$RepoPath/model.onnx?download=true" -OutputPath "$OutputDir\model.onnx"

# Download MatMul files
$matMulRanges = @(
    8851..8853
    8878..8884
    8909..8915
    8940..8946
    8971..8977
    9002..9008
    9033..9039
    9064..9070
    9095..9101
    9126..9132
    9157..9163
    9188..9194
    9219..9225
    9250..9256
    9281..9287
    9312..9318
    9343..9349
    9374..9380
    9405..9411
    9436..9442
    9467..9473
    9498..9504
    9529..9535
    9560..9566
    9591..9597
    9622..9628
    9653..9659
    9684..9690
    9715..9718
    9722
)

$matMulRanges | ForEach-Object {
    if ($_ -is [array]) {
        $_ | ForEach-Object {
            Download-FileWithProgress -Url "$BaseUrl/$RepoPath/onnx__MatMul_$_?download=true" -OutputPath "$OutputDir\onnx__MatMul_$_"
        }
    } else {
        Download-FileWithProgress -Url "$BaseUrl/$RepoPath/onnx__MatMul_$_?download=true" -OutputPath "$OutputDir\onnx__MatMul_$_"
    }
}

# Download tokenizer files
Download-FileWithProgress -Url "$BaseUrl/$RepoPath/special_tokens_map.json?download=true" -OutputPath "$OutputDir\special_tokens_map.json"
Download-FileWithProgress -Url "$BaseUrl/$RepoPath/tokenizer.json?download=true" -OutputPath "$OutputDir\tokenizer.json"
Download-FileWithProgress -Url "$BaseUrl/$RepoPath/tokenizer_config.json?download=true" -OutputPath "$OutputDir\tokenizer_config.json"

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host "Download script completed!" -ForegroundColor Green
Write-Host "Files downloaded to: $OutputDir" -ForegroundColor Green
Write-Host "Total time taken: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green