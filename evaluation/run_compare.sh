# Full evaluation on both tracks (~40-60 minutes)
  # python compare_base_vs_finetuned.py \
  #     --base-model Qwen/Qwen3-8B \
  #     --lora-path ../finetuning/models/qwen3-8b-lora \
  #     --track both
  python compare_base_vs_finetuned.py \
      --base-model meta-llama/Llama-3.1-8B-Instruct \
      --lora-path ../finetuning/models/meta-llama-llama-3.1-8b-instruct-lora \
      --track both


  # python compare_base_vs_finetuned.py \
  #     --base-model Qwen/Qwen3-1.7B \
  #     --lora-path ../finetuning/models/qwen3-1.7b-lora \
  #     --track both
  
    # python compare_base_vs_finetuned.py \
    #   --base-model Qwen/Qwen3-4B-Instruct-2507 \
    #   --lora-path ../finetuning/models/qwen3-4b-instruct-2507-lora \
    #   --track both