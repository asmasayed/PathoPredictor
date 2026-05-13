"""
Module 1 - Step 4: Train DNABERT-style masked language model on H5N1 Arrow Dataset.
Optimized for 6GB VRAM GPUs using Mixed Precision and Gradient Accumulation.
"""
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

def train_dnbert(dataset_dir: str, model_output_dir: str, base_model_name: str = "zhihan1996/DNA_bert_6") -> None:
    print("\n--- Initializing VRAM-Optimized Training ---")
    
    # 1. Hardware Check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True # Slight speedup for Ampere GPUs (like the RTX 40 series)

    # 2. Load the Arrow Dataset from disk
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}. Run tokenization first.")
    
    print(f"Loading Arrow dataset from {dataset_dir}...")
    dataset = load_from_disk(dataset_dir)
    print(f"Loaded {len(dataset)} training sequences.")

    # 3. Load Model and Tokenizer
    print(f"Loading base model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForMaskedLM.from_pretrained(base_model_name)

    # 4. Data Collator for Masked Language Modeling
    # This dynamically masks 15% of the tokens in a batch for the model to predict
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # 5. Training Arguments (Tuned for ~6GB VRAM)
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=3,
        
        # VRAM Optimization Core:
        per_device_train_batch_size=4,        # Keep this small so we don't OOM
        gradient_accumulation_steps=8,        # Accumulate gradients over 8 steps (Effective Batch Size = 32)
        fp16=True,                            # Mixed precision (half-precision floats) halves memory usage
        
        learning_rate=5e-5,
        weight_decay=0.01,
        save_strategy="epoch",                # Save a checkpoint at the end of each epoch
        logging_steps=50,
        dataloader_num_workers=4,             # Keep data feeding fast
        remove_unused_columns=False,          # Keep this False for custom HF datasets
        report_to="none"                      # Disable wandb/tensorboard logging for now to keep it clean
    )

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 7. Start Training
    print("\nStarting Masked Language Model Fine-Tuning...")
    trainer.train()

    # 8. Save Final Model
    print(f"\nSaving final model and tokenizer to {model_output_dir}...")
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print("Training complete! Your fine-tuned DNABERT model is ready.")

if __name__ == "__main__":
    from pathlib import Path
    project_root = Path(__file__).resolve().parents[2]
    
    in_dir = project_root / "data" / "processed" / "module1" / "hf_tokenized_dataset"
    out_dir = project_root / "models" / "module1_dnbert"
    
    train_dnbert(str(in_dir), str(out_dir))