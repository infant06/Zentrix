$files = Get-ChildItem -Path "C:\AiResearch\Zenllm\crates\zen-core\src" -Filter "*.rs" -Recurse

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw
    
    # Replace in normal_loaders.rs
    if ($file.Name -eq "normal_loaders.rs") {
        $content = $content -replace 'todo\!\(\)', 'candle_core::bail!("X-LoRA is not supported for this model type")'
    }
    
    # Replace in models directory
    if ($file.FullName -match "models\\") {
        $content = $content -replace 'unimplemented\!\(\)', 'candle_core::bail!("X-LoRA is not supported for this model")'
    }

    # Write back if changed
    Set-Content -Path $file.FullName -Value $content -NoNewline
}

Write-Output "Done cleaning up unimplemented and todo macros in models and loaders."
