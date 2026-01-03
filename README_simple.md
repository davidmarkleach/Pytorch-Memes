🎭 Initial commit: PyTorch Meme Generator

A meme generator that uses AI to create funny memes with classic Impact font styling.

Features:
- Fine-tuned GPT-2 model for meme text generation
- OpenAI API integration for high-quality text (GPT-4o-mini)
- 35+ popular meme templates from Imgflip API
- Streamlit web UI with batch generation mode
- Classic meme styling: white Impact font with black outline

Scripts:
- download_memes.py: Fetch templates from Imgflip API
- train_meme_model.py: Fine-tune GPT-2 on meme data
- generate_memes.py: Core meme generation logic
- app.py: Streamlit web interface

Stack: Python, PyTorch, Transformers, PIL, Streamlit, OpenAI