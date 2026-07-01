Write-Host "1. Building zenllm in Release Mode with lance-storage..."
cargo build --release --bin zenllm --features lance-storage

Write-Host "
2. Testing Lance Vector DB Storage..."
.\target\release\zenllm.exe vector add -c test_collection -i test_id -v 0.1,0.2,0.3 -m '{"key":"value"}'

Write-Host "
3. Testing CPU-Only Mode..."
.\target\release\zenllm.exe run -m "C:\AiResearch\Zenllm\models\qwen2.5-0.5b-instruct-q4_0.gguf" --mode cpu-only -i "Say hello"

Write-Host "
4. Testing Partial Offload Mode..."
.\target\release\zenllm.exe run -m "C:\AiResearch\Zenllm\models\qwen2.5-0.5b-instruct-q4_0.gguf" --mode partial-offload -i "Say hello"

Write-Host "
5. Testing FlexLoad Mode..."
.\target\release\zenllm.exe run -m "C:\AiResearch\Zenllm\models\qwen2.5-0.5b-instruct-q4_0.gguf" --mode flex-load -i "Say hello"

Write-Host "
6. Testing Multi-GPU Mode..."
.\target\release\zenllm.exe run -m "C:\AiResearch\Zenllm\models\qwen2.5-0.5b-instruct-q4_0.gguf" --mode multi-gpu -i "Say hello"
