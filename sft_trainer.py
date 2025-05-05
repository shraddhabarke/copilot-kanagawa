import json, os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

torch.cuda.empty_cache()
torch.cuda.ipc_collect()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
def save_jsonl(dataset, path):
    """Save HuggingFace dataset to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for example in dataset:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")

def main():
    # --- Step 1: Load and split dataset ---
    full_dataset = load_dataset("json", data_files="sft_sandpiper.jsonl", split="train")

    # Shuffle and split 90% train, 10% test
    full_dataset = full_dataset.shuffle(seed=42)
    split_dataset = full_dataset.train_test_split(test_size=0.1)

    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    # --- Step 2: Save splits into separate JSONL files ---
    save_jsonl(train_dataset, "train_sft_dataset.jsonl")
    save_jsonl(eval_dataset, "test_sft_dataset.jsonl")
    print(f"✅ Saved {len(train_dataset)} train examples and {len(eval_dataset)} test examples.")

    # --- Step 3: Reload properly using HuggingFace data_files ---
    datasets = load_dataset(
        "json",
        data_files={
            "train": "train_sft_dataset.jsonl",
            "test": "test_sft_dataset.jsonl"
        }
    )
    train_dataset = datasets["train"]
    eval_dataset = datasets["test"]

    # --- Step 4: Load model and tokenizer ---
    model_id = "microsoft/Phi-3-mini-128k-instruct" #phi-3-mini-4k-instruct"  # or "microsoft/phi-4" later
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2")
    #model = model.to("cuda")  # Move model to GPU

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"  # Safe for FlashAttention + bf16

    # --- Step 5: Define formatting function ---
    def formatting_func(example):
        prompt = example["prompt"]
        completion = example["completion"]
        if not isinstance(prompt, str):
            prompt = " ".join(prompt)  # Assume prompt is a list of words
        if not isinstance(completion, str):
            completion = " ".join(completion)
        return [prompt.strip() + "\n\n" + completion.strip()]

    # --- Step 6: Define training configuration ---
    training_args = SFTConfig(
        output_dir=f"{model_id.replace('/', '_')}-vector-SFT",
        logging_steps=10,
        bf16=True,
        use_liger_kernel=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        dataset_num_proc=32,
        num_train_epochs=30,  # 🔥 adjust higher if needed
        #evaluation_strategy="steps",
        #eval_steps=50,
        #save_steps=100,
        #save_total_limit=2,
        logging_dir="./logs",
        #report_to="none",  # no wandb logging
    )

    # --- Step 7: Initialize Trainer ---
    trainer = SFTTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        processing_class=tokenizer,  # ✅ newer TRL API
    )

    # --- Step 8: Start training ---
    trainer.train()

if __name__ == "__main__":
    main()

