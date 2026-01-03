# 🎭 PyTorch Meme Generator

Generate memes with AI-powered text using GPT-2 fine-tuned on popular meme formats!

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Meme Templates

```bash
python download_memes.py
```

This downloads popular meme templates from Imgflip to the `meme_templates/` folder.

### 3. Train the Model

```bash
python train_meme_model.py
```

This fine-tunes GPT-2 on the meme text data (~10 epochs, takes a few minutes).

### 4. Generate Memes!

```bash
# Generate a single Drake meme
python generate_memes.py --meme "Drake"

# Generate with custom text
python generate_memes.py --meme "Drake" --top "Using print to debug" --bottom "Using a debugger"

# Generate a batch of memes
python generate_memes.py --batch

# List all available meme formats
python generate_memes.py --list
```

## 📁 Project Structure

```
Pytorch-Memes/
├── meme_templates/          # Downloaded meme template images
├── generated_memes/         # Output folder for generated memes
├── meme_model_final/        # Trained GPT-2 model (after training)
├── meme_data.txt           # Training data (40+ meme formats)
├── download_memes.py       # Download meme templates
├── train_meme_model.py     # Fine-tune GPT-2
├── generate_memes.py       # Generate memes with images
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎨 Supported Meme Formats

The generator supports 30+ popular meme formats including:

- **Drake Hotline Bling** - Rejecting vs. Approving format
- **Distracted Boyfriend** - Something more appealing catches attention
- **One Does Not Simply** - Boromir explaining difficulty
- **Surprised Pikachu** - Obvious consequence reaction
- **Two Buttons** - Difficult choice
- **Woman Yelling at Cat** - Argument format
- **Expanding Brain** - Increasing intelligence/absurdity
- **This Is Fine** - Everything is on fire
- **Stonks** - Financial gains meme
- **Change My Mind** - Debate format
- And many more!

## 💻 Usage Examples

### Generate with AI Text

```python
from generate_memes import MemeGenerator

generator = MemeGenerator()

# Generate Drake meme with AI text
generator.generate_meme("Drake")

# Generate multiple variations
generator.generate_batch(["Drake", "Distracted Boyfriend"], count_per_meme=5)
```

### Generate with Custom Text

```python
generator.generate_meme(
    "One Does Not Simply",
    custom_text={
        "top": "One does not simply",
        "bottom": "train a neural network on the first try"
    }
)
```

### Command Line Options

```bash
# Single meme with AI-generated text
python generate_memes.py --meme "Surprised Pikachu"

# Custom text
python generate_memes.py --meme "Drake" --top "Tabs" --bottom "Spaces"

# Batch generation (3 variations each of Drake, Distracted Boyfriend, One Does Not Simply)
python generate_memes.py --batch --count 5

# Adjust creativity (higher = more random)
python generate_memes.py --meme "Drake" --temperature 1.0

# List available formats
python generate_memes.py --list
```

## 🧠 How It Works

1. **Training Data** (`meme_data.txt`): Contains 40+ meme formats with example text in the format:
   ```
   [Meme Name] Top: text | Bottom: text
   ```

2. **Model Training**: Fine-tunes GPT-2 on the meme data to learn meme-style humor and format patterns.

3. **Text Generation**: Given a meme format prompt, generates contextually appropriate meme text.

4. **Image Composition**: Uses PIL to overlay the generated text on meme templates with classic white Impact font and black outline.

## ⚙️ Training Options

```bash
# Train with more epochs
python train_meme_model.py --epochs 20

# Test existing model
python train_meme_model.py --test
```

## 📝 Meme Data Format

The training data follows this format:

```
[Meme Name] Top: top text | Bottom: bottom text
```

For memes with more panels:
```
[Meme Name] Top: text | Middle: text | Bottom: text
```

Feel free to add your own meme entries to `meme_data.txt` and retrain!

## 🔧 Troubleshooting

### "Model not found"
Run `python train_meme_model.py` first to train the model.

### "Template file not found"
Run `python download_memes.py` to download meme templates.

### Font issues
The generator tries to use Impact font. If not available, it falls back to Arial or default font. For best results on Linux:
```bash
sudo apt install ttf-mscorefonts-installer
```

### CUDA out of memory
Reduce batch size in `train_meme_model.py` or use CPU training.

## 📄 License

MIT License - feel free to use, modify, and distribute!

## 🤝 Contributing

1. Add new meme formats to `meme_data.txt`
2. Add template mappings in `generate_memes.py`
3. Improve text positioning for new formats

---

Made with ❤️ and PyTorch

