#!/usr/bin/env python3
"""
Train Meme Generator Model
Fine-tunes GPT-2 on meme_data.txt to generate meme text variations
"""

import os
import torch
from pathlib import Path
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# Paths
BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "meme_data.txt"
OUTPUT_DIR = BASE_DIR / "meme_model_final"

# Training hyperparameters
EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
BLOCK_SIZE = 128
WARMUP_STEPS = 100
SAVE_STEPS = 500


def load_and_prepare_data(tokenizer, file_path: Path, block_size: int = BLOCK_SIZE):
    """Load training data and prepare for model."""
    print(f"📖 Loading data from {file_path}")

    # Read and filter the data file
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Filter out comments and empty lines, keep meme entries
    meme_lines = []
    for line in lines:
        line = line.strip()
        # Keep lines that start with [ which indicates a meme entry
        if line.startswith("["):
            meme_lines.append(line)

    # Create a temporary clean file for training
    clean_file = BASE_DIR / "meme_data_clean.txt"
    with open(clean_file, "w", encoding="utf-8") as f:
        f.write("\n".join(meme_lines))

    print(f"✅ Found {len(meme_lines)} meme entries")

    # Create dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=str(clean_file),
        block_size=block_size,
    )

    return dataset


def train_model():
    """Main training function."""
    print("🎭 Meme Generator Model Training")
    print("=" * 50)

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")

    # Check if data file exists
    if not DATA_FILE.exists():
        print(f"❌ Data file not found: {DATA_FILE}")
        print("Please ensure meme_data.txt exists in the project directory.")
        return

    # Load tokenizer and model
    print("\n📥 Loading GPT-2 model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Set pad token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Move model to device
    model.to(device)
    print(f"✅ Model loaded: {model.num_parameters():,} parameters")

    # Load data
    train_dataset = load_and_prepare_data(tokenizer, DATA_FILE)

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 is causal LM, not masked LM
    )

    # Training arguments
    print("\n⚙️  Setting up training...")
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        logging_steps=50,
        logging_dir=str(BASE_DIR / "logs"),
        prediction_loss_only=True,
        # Disable features that might cause issues
        fp16=False,  # Set to True if you have a compatible GPU
        dataloader_num_workers=0,
        # Progress
        disable_tqdm=False,
        report_to="none",  # Disable wandb/tensorboard
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Train!
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    print("-" * 50)

    trainer.train()

    # Save the final model
    print("\n💾 Saving model...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"✅ Model saved to: {OUTPUT_DIR}")
    print("\n🎉 Training complete!")

    # Clean up temporary file
    clean_file = BASE_DIR / "meme_data_clean.txt"
    if clean_file.exists():
        clean_file.unlink()

    return model, tokenizer


def generate_sample(model, tokenizer, prompt: str, max_length: int = 100):
    """Generate a sample meme text."""
    model.eval()
    device = next(model.parameters()).device

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def test_model():
    """Test the trained model with sample generations."""
    print("\n🧪 Testing trained model...")
    print("-" * 50)

    if not OUTPUT_DIR.exists():
        print(f"❌ Model not found at {OUTPUT_DIR}")
        print("Please run training first.")
        return

    # Load model
    tokenizer = GPT2Tokenizer.from_pretrained(str(OUTPUT_DIR))
    model = GPT2LMHeadModel.from_pretrained(str(OUTPUT_DIR))

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    # Test prompts
    test_prompts = [
        "[Drake] Top:",
        "[One Does Not Simply] Top:",
        "[Distracted Boyfriend] Top:",
        "[Surprised Pikachu] Top:",
    ]

    print("\n📝 Sample generations:")
    for prompt in test_prompts:
        generated = generate_sample(model, tokenizer, prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train meme generator model")
    parser.add_argument("--test", action="store_true", help="Test existing model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")

    args = parser.parse_args()

    if args.test:
        test_model()
    else:
        EPOCHS = args.epochs
        model, tokenizer = train_model()
        test_model()

