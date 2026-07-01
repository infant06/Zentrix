import zentrix
import json

def test_hardware_detection():
    print("--- Testing Hardware Detection ---")
    hw_info_str = zentrix.detect_hardware()
    hw_info = json.loads(hw_info_str)
    print(json.dumps(hw_info, indent=2))
    print()
    return hw_info

def test_runner_mode(model_path: str, mode: str):
    print(f"--- Testing Runner in '{mode}' mode ---")
    try:
        which = zentrix.Which.GGUF(
            tok_model_id=None,
            quantized_model_id=None,
            quantized_filename=model_path,
            tokenizer_json=None,
            topology=None
        )
        
        runner = zentrix.Runner(
            which=which,
            mode=mode
        )
        
        request = zentrix.ChatCompletionRequest(
            messages=[{"role": "user", "content": "Explain what a GPU is briefly."}],
            model="test",
            max_tokens=50
        )
        
        response = runner.send_chat_completion_request(request)
        print("Success! Response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Failed to run in mode '{mode}': {e}")
    print()

if __name__ == "__main__":
    hw_info = test_hardware_detection()
    
    # Use a small test model available locally
    # Update this path if the model is located elsewhere
    model_path = r"C:\AiResearch\Zenllm\models\qwen2.5-0.5b-instruct-q4_0.gguf"
    
    modes_to_test = ["cpu-only", "partial-offload", "gpu-only", "flexload"]
    
    for mode in modes_to_test:
        test_runner_mode(model_path, mode)
