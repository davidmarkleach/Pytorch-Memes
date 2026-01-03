#!/usr/bin/env python3
"""
=============================================================================
MEME GENERATOR WITH AI TEXT GENERATION
=============================================================================

This script creates memes by:
1. Generating funny text using AI (either OpenAI API or a local GPT-2 model)
2. Loading a meme template image (like Drake, Distracted Boyfriend, etc.)
3. Drawing the text on the image with classic meme styling (white Impact font
   with black outline)
4. Saving the final meme image

You can use this script in three ways:
- Command line: python generate_memes.py --meme "Drake" --top "Text" --bottom "Text"
- Import in Python: from generate_memes import MemeGenerator
- Through the Streamlit web app: streamlit run app.py

Dependencies:
- torch: PyTorch for running the local GPT-2 model
- PIL (Pillow): For image manipulation
- transformers: Hugging Face library for GPT-2
- openai (optional): For using OpenAI's API for better text generation
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os          # For file path operations and environment variables
import re          # For regular expressions (parsing text)
import random      # For random selection (not heavily used here)
import textwrap    # For wrapping long text (not heavily used here)
from pathlib import Path  # Modern way to handle file paths
from typing import Optional, Tuple, Dict, List  # Type hints for better code clarity

# PyTorch - the deep learning framework
import torch

# PIL (Python Imaging Library) - for loading, editing, and saving images
from PIL import Image, ImageDraw, ImageFont

# Hugging Face Transformers - for loading and using GPT-2
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# =============================================================================
# OPTIONAL: OpenAI Support
# =============================================================================
# We try to import OpenAI, but it's optional. If not installed, the code
# still works but you can only use the local GPT-2 model.

try:
    from openai import OpenAI  # OpenAI's official Python client
    OPENAI_AVAILABLE = True    # Flag to check if OpenAI is available
except ImportError:
    OPENAI_AVAILABLE = False   # OpenAI not installed, that's okay!

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# These paths tell the script where to find templates and save outputs.
# Path(__file__).parent gets the folder where this script is located.

BASE_DIR = Path(__file__).parent              # Root project folder
MODEL_DIR = BASE_DIR / "meme_model_final"     # Where the trained GPT-2 model lives
TEMPLATES_DIR = BASE_DIR / "meme_templates"   # Where meme template images are stored
OUTPUT_DIR = BASE_DIR / "generated_memes"     # Where generated memes are saved

# Create the output directory if it doesn't exist
# exist_ok=True means "don't error if it already exists"
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# MEME TEMPLATE MAPPING
# =============================================================================
# This dictionary maps human-readable meme names to their image filenames.
# The filenames match what we downloaded from the Imgflip API.
#
# Example: "Drake" -> "drake_hotline_bling.jpg"
# This lets users say --meme "Drake" instead of remembering the exact filename.

MEME_TEMPLATES = {
    "Drake": "drake_hotline_bling.jpg",
    "Distracted Boyfriend": "distracted_boyfriend.jpg",
    "One Does Not Simply": "one_does_not_simply.jpg",
    "Surprised Pikachu": "surprised_pikachu.jpg",
    "Two Buttons": "two_buttons.jpg",
    "Woman Yelling at Cat": "woman_yelling_at_cat.jpg",
    "Expanding Brain": "expanding_brain.jpg",
    "Is This a Pigeon": "is_this_a_pigeon.jpg",
    "This Is Fine": "this_is_fine.jpg",
    "Stonks": "stonks.jpg",
    "Change My Mind": "change_my_mind.jpg",
    "Always Has Been": "always_has_been.jpg",
    "Bernie Sitting": "bernie_i_am_once_again_asking_for_your_support.jpg",
    "Exit 12": "left_exit_12_off_ramp.jpg",
    "Sleeping Shaq": "sleeping_shaq.jpg",
    "They're The Same Picture": "theyre_the_same_picture.jpg",
    "Batman Slapping Robin": "batman_slapping_robin.jpg",
    "Roll Safe": "roll_safe_think_about_it.jpg",
    "Gru's Plan": "grus_plan.jpg",
    "SpongeBob Mocking": "mocking_spongebob.jpg",
    "Hide the Pain Harold": "hide_the_pain_harold.jpg",
    "Success Kid": "success_kid.jpg",
    "First Time": "first_time.jpg",
    "Tuxedo Pooh": "tuxedo_winnie_the_pooh.jpg",
    "Ancient Aliens": "ancient_aliens.jpg",
    "Futurama Fry": "futurama_fry.jpg",
    "Flex Tape": "flex_tape.jpg",
    "Confused Math Lady": "confused_math_lady.jpg",
    "Panik Kalm Panik": "panik_kalm_panik.jpg",
    "Trade Offer": "trade_offer.jpg",
    "Sad Pablo Escobar": "sad_pablo_escobar.jpg",
    "Epic Handshake": "epic_handshake.jpg",
    "UNO Draw 25": "uno_draw_25_cards.jpg",
    "Bike Fall": "bike_fall.jpg",
}

# =============================================================================
# TEXT POSITIONING CONFIGURATION
# =============================================================================
# Different memes have text in different places! Drake has text on the right,
# One Does Not Simply has text at top and bottom, etc.
#
# This dictionary defines WHERE to place text for each meme template.
#
# Format:
#   "template_name": {
#       "top": {"pos": (x_ratio, y_ratio), "max_width": width_ratio},
#       "bottom": {"pos": (x_ratio, y_ratio), "max_width": width_ratio},
#   }
#
# The values are RATIOS (0.0 to 1.0) of the image dimensions:
#   - pos: (0.5, 0.1) means "50% from left, 10% from top" (centered near top)
#   - max_width: 0.9 means "text can be up to 90% of image width"

MEME_TEXT_POSITIONS = {
    # Drake: Text goes on the RIGHT side (where Drake is pointing/approving)
    "drake_hotline_bling": {
        "top": {"pos": (0.73, 0.25), "max_width": 0.45},     # Right side, upper panel
        "bottom": {"pos": (0.73, 0.75), "max_width": 0.45},  # Right side, lower panel
    },

    # Distracted Boyfriend: Labels go at the BOTTOM of each person
    "distracted_boyfriend": {
        "top": {"pos": (0.15, 0.85), "max_width": 0.25},     # Girlfriend (left)
        "middle": {"pos": (0.5, 0.85), "max_width": 0.25},   # Other woman (center)
        "bottom": {"pos": (0.82, 0.85), "max_width": 0.25},  # Boyfriend (right)
    },

    # One Does Not Simply: Classic top/bottom format
    "one_does_not_simply": {
        "top": {"pos": (0.5, 0.1), "max_width": 0.9},    # Centered at top
        "bottom": {"pos": (0.5, 0.9), "max_width": 0.9}, # Centered at bottom
    },

    # Two Buttons: Text goes on the buttons (left side)
    "two_buttons": {
        "top": {"pos": (0.25, 0.2), "max_width": 0.2},    # Upper button
        "bottom": {"pos": (0.25, 0.45), "max_width": 0.2}, # Lower button
    },

    # Expanding Brain: Four panels, text on the left
    "expanding_brain": {
        "top": {"pos": (0.25, 0.12), "max_width": 0.45},    # First panel
        "middle": {"pos": (0.25, 0.38), "max_width": 0.45}, # Second panel
        "bottom": {"pos": (0.25, 0.62), "max_width": 0.45}, # Third panel
        "final": {"pos": (0.25, 0.88), "max_width": 0.45},  # Fourth panel (galaxy brain)
    },

    # Change My Mind: Text goes on the sign
    "change_my_mind": {
        "bottom": {"pos": (0.55, 0.75), "max_width": 0.5},  # On the sign
    },

    # Gru's Plan: Four panels
    "grus_plan": {
        "top": {"pos": (0.73, 0.12), "max_width": 0.45},    # Panel 1
        "middle": {"pos": (0.73, 0.38), "max_width": 0.45}, # Panel 2
        "bottom": {"pos": (0.25, 0.62), "max_width": 0.45}, # Panel 3 (the twist)
        "final": {"pos": (0.73, 0.88), "max_width": 0.45},  # Panel 4 (realization)
    },

    # Default: If we don't have specific positions, use classic top/bottom
    "default": {
        "top": {"pos": (0.5, 0.1), "max_width": 0.9},
        "bottom": {"pos": (0.5, 0.9), "max_width": 0.9},
    },
}


# =============================================================================
# MEME GENERATOR CLASS
# =============================================================================
# This is the main class that does all the work. It's organized as a class
# so we can:
#   1. Load the AI model once and reuse it (faster)
#   2. Keep settings organized
#   3. Cache fonts for performance

class MemeGenerator:
    """
    Main class for generating memes with AI-powered text.

    This class handles:
    - Loading and running AI models (GPT-2 or OpenAI)
    - Loading meme template images
    - Drawing text on images with proper styling
    - Saving the final memes

    Example usage:
        generator = MemeGenerator()
        generator.generate_meme("Drake", custom_text={"top": "Bad thing", "bottom": "Good thing"})
    """

    def __init__(self, model_path: Path = MODEL_DIR, openai_api_key: Optional[str] = None):
        """
        Initialize the meme generator.

        Args:
            model_path: Where to find the trained GPT-2 model (default: meme_model_final/)
            openai_api_key: Optional OpenAI API key for better text generation
        """
        # These will hold the GPT-2 model and tokenizer once loaded
        # We set them to None initially and load them lazily (only when needed)
        self.model = None
        self.tokenizer = None

        # Detect the best available device for running the AI model:
        # - "cuda": NVIDIA GPU (fastest)
        # - "mps": Apple Silicon GPU (fast on M1/M2 Macs)
        # - "cpu": Regular CPU (slowest but always works)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model_path = model_path

        # Cache for loaded fonts (loading fonts is slow, so we cache them)
        # Key: font size (int), Value: loaded font object
        self.font_cache: Dict[int, ImageFont.FreeTypeFont] = {}

        # OpenAI client (for using GPT-4 instead of local GPT-2)
        self.openai_client = None

        # Set up OpenAI if a key was provided or exists in environment
        if openai_api_key:
            self.set_openai_key(openai_api_key)
        elif os.environ.get("OPENAI_API_KEY"):
            # Check for API key in environment variable
            self.set_openai_key(os.environ["OPENAI_API_KEY"])

    def set_openai_key(self, api_key: str):
        """
        Configure the OpenAI API client.

        Args:
            api_key: Your OpenAI API key (starts with "sk-")
        """
        if OPENAI_AVAILABLE:
            # Create the OpenAI client with the provided key
            self.openai_client = OpenAI(api_key=api_key)
        else:
            print("⚠️  OpenAI package not installed. Run: pip install openai")

    def load_model(self):
        """
        Load the fine-tuned GPT-2 model from disk.

        This is called automatically when you try to generate text with the
        local model. We load it "lazily" (only when needed) because:
        1. Loading takes a few seconds
        2. You might only want to use OpenAI, not the local model

        The model consists of two parts:
        - Tokenizer: Converts text to numbers (tokens) that the model understands
        - Model: The actual neural network that generates text
        """
        print(f"📥 Loading model from {self.model_path}")

        # Check if the model folder exists
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please run train_meme_model.py first."
            )

        # Load the tokenizer (converts text <-> tokens)
        # from_pretrained() loads from a saved model directory
        self.tokenizer = GPT2Tokenizer.from_pretrained(str(self.model_path))

        # Load the actual GPT-2 model weights
        self.model = GPT2LMHeadModel.from_pretrained(str(self.model_path))

        # Move the model to the best available device (GPU if available)
        self.model.to(self.device)

        # Set model to evaluation mode (disables training-specific behaviors)
        self.model.eval()

        print(f"✅ Model loaded on {self.device}")

    def generate_meme_text(
        self,
        meme_name: str,
        temperature: float = 0.8,
        max_length: int = 100,
    ) -> str:
        """
        Generate meme text using the LOCAL fine-tuned GPT-2 model.

        This uses the model we trained on meme_data.txt. It's fast and free,
        but the quality is lower than OpenAI because:
        - GPT-2 is a smaller, older model
        - We only had ~100 training examples

        Args:
            meme_name: Which meme format (e.g., "Drake")
            temperature: Controls randomness (0.0 = predictable, 1.0+ = creative/chaotic)
            max_length: Maximum tokens to generate

        Returns:
            Raw generated text like "[Drake] Top: something | Bottom: something else"
        """
        # Load the model if we haven't already
        if self.model is None:
            self.load_model()

        # Create a prompt in the format the model was trained on
        # The model learned to complete text that starts like this
        prompt = f"[{meme_name}] Top:"

        # Convert the text prompt into token IDs (numbers the model understands)
        # return_tensors="pt" means return a PyTorch tensor
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # Generate text! This is where the magic happens.
        # torch.no_grad() tells PyTorch we're not training, just generating
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,                    # Our prompt as tokens
                max_length=max_length,     # Don't generate more than this many tokens
                num_return_sequences=1,    # Just generate one completion
                temperature=temperature,   # Randomness control
                top_k=50,                  # Only consider top 50 most likely tokens
                top_p=0.95,                # Nucleus sampling (advanced randomness control)
                do_sample=True,            # Enable random sampling (not just most likely)
                pad_token_id=self.tokenizer.eos_token_id,  # Handle padding
            )

        # Convert the generated tokens back into human-readable text
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated

    def generate_meme_text_openai(
        self,
        meme_name: str,
        topic: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ) -> Dict[str, str]:
        """
        Generate meme text using OpenAI's API (GPT-4).

        This produces MUCH better results than the local model because:
        - GPT-4 is vastly more capable
        - It understands humor and context
        - We give it detailed instructions about each meme format

        Args:
            meme_name: Which meme format (e.g., "Drake")
            topic: Optional topic hint (e.g., "programming", "coffee")
            model: Which OpenAI model to use (gpt-4o-mini is fast and cheap)

        Returns:
            Dictionary like {"top": "text", "bottom": "text", "meme_name": "Drake"}
        """
        # Make sure we have an OpenAI client configured
        if not self.openai_client:
            raise ValueError("OpenAI API key not set. Use set_openai_key() or set OPENAI_API_KEY env var.")

        # =================================================================
        # MEME FORMAT DESCRIPTIONS
        # =================================================================
        # These descriptions help GPT-4 understand HOW each meme format works.
        # Without this, it might not know that Drake = "reject vs prefer" format.

        meme_formats = {
            "Drake": "Two panels: top panel shows rejection/dislike, bottom panel shows approval/preference. Format: thing you reject vs thing you prefer.",
            "Distracted Boyfriend": "Three people: boyfriend (you/someone), girlfriend (current thing), other woman (tempting new thing). Shows being distracted by something new.",
            "One Does Not Simply": "Boromir saying 'One does not simply...' followed by something difficult. Classic difficulty/impossibility format.",
            "Surprised Pikachu": "Setup an obvious action, then show surprised reaction to the obvious consequence.",
            "Two Buttons": "Two options that are both appealing but you can only pick one. Shows difficult choice.",
            "Change My Mind": "Hot take or controversial opinion with 'Change my mind' - debate format.",
            "Expanding Brain": "Four levels from simple to galaxy brain - increasingly absurd or 'enlightened' options.",
            "Batman Slapping Robin": "Robin says something wrong, Batman slaps and corrects. Shutting down bad takes.",
            "Roll Safe": "Pointing to head meme - 'Can't have X problem if you Y' - ironic life hack logic.",
            "Gru's Plan": "Four panels: 1) plan step 1, 2) plan step 2, 3) unexpected bad outcome, 4) realization of bad outcome.",
            "Trade Offer": "I receive: X, You receive: Y - usually unfair or funny trade.",
            "Panik Kalm Panik": "Three panels: Panic at problem, calm at solution, panic again when solution backfires.",
            "Epic Handshake": "Two opposing things agreeing on one common thing they both share.",
        }

        # Get the description for this meme, or use a generic one
        format_desc = meme_formats.get(meme_name, "Classic top text / bottom text meme format.")

        # Add topic hint to the prompt if provided
        if topic:
            topic_hint = f" Make it about: {topic}"
        else:
            topic_hint = " Pick a funny relatable topic (tech, work, life, gaming, etc)."

        # =================================================================
        # BUILD THE PROMPT FOR GPT-4
        # =================================================================
        # We ask GPT-4 to return JSON so we can easily parse the response

        prompt = f"""Generate a funny meme for the "{meme_name}" format.

Format description: {format_desc}
{topic_hint}

Respond with ONLY a JSON object (no markdown, no explanation):
{{"top": "top text here", "bottom": "bottom text here"}}

For multi-panel memes like Distracted Boyfriend, use:
{{"top": "girlfriend label", "middle": "other woman label", "bottom": "boyfriend label"}}

Keep text SHORT and punchy (under 10 words per field). Be actually funny, not generic."""

        try:
            # =================================================================
            # CALL THE OPENAI API
            # =================================================================
            response = self.openai_client.chat.completions.create(
                model=model,  # e.g., "gpt-4o-mini"
                messages=[
                    # System message sets the AI's personality/role
                    {
                        "role": "system",
                        "content": "You are a meme expert who creates hilarious, relatable memes. You respond only with JSON, no markdown."
                    },
                    # User message is our actual request
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.9,    # High temperature = more creative
                max_tokens=150,     # Limit response length
            )

            # =================================================================
            # PARSE THE RESPONSE
            # =================================================================
            # Extract the text content from the API response
            content = response.choices[0].message.content.strip()

            # Sometimes GPT wraps JSON in markdown code blocks, remove them
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]  # Remove "json" language tag
            content = content.strip()

            # Parse the JSON string into a Python dictionary
            import json
            result = json.loads(content)

            # Add the meme name to the result
            result["meme_name"] = meme_name
            return result

        except Exception as e:
            # If anything goes wrong, return an error message as the meme text
            print(f"❌ OpenAI error: {e}")
            return {"meme_name": meme_name, "top": "Error generating", "bottom": str(e)[:30]}

    def parse_meme_text(self, generated_text: str) -> Dict[str, str]:
        """
        Parse text generated by the LOCAL GPT-2 model into usable components.

        The local model generates raw text like:
            "[Drake] Top: bad thing | Bottom: good thing"

        This function extracts the parts into a dictionary:
            {"meme_name": "Drake", "top": "bad thing", "bottom": "good thing"}

        Args:
            generated_text: Raw text from generate_meme_text()

        Returns:
            Dictionary with parsed text fields
        """
        # Initialize result with empty strings for all possible fields
        result = {"meme_name": "", "top": "", "bottom": "", "middle": "", "caption": ""}

        # Extract meme name from [brackets] using regex
        # r"\[([^\]]+)\]" means: find [ followed by any chars except ], then ]
        name_match = re.search(r"\[([^\]]+)\]", generated_text)
        if name_match:
            result["meme_name"] = name_match.group(1)  # group(1) = the part in parentheses

        # Get everything after the [Meme Name] part
        text_part = generated_text.split("]")[-1].strip()

        # Split by | to separate different text positions
        # "Top: hello | Bottom: world" -> ["Top: hello", "Bottom: world"]
        parts = text_part.split("|")

        # Parse each part to find top/bottom/middle/caption
        for part in parts:
            part = part.strip()
            if part.lower().startswith("top:"):
                result["top"] = part[4:].strip()      # Remove "top:" prefix
            elif part.lower().startswith("bottom:"):
                result["bottom"] = part[7:].strip()   # Remove "bottom:" prefix
            elif part.lower().startswith("middle:"):
                result["middle"] = part[7:].strip()
            elif part.lower().startswith("caption:"):
                result["caption"] = part[8:].strip()

        return result

    def get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """
        Get the Impact font at a specific size, with caching for performance.

        Impact is THE classic meme font - bold, blocky, and readable.

        Args:
            size: Font size in pixels

        Returns:
            A PIL font object ready for drawing
        """
        # Check cache first (loading fonts is slow)
        if size in self.font_cache:
            return self.font_cache[size]

        # List of places to look for the Impact font on different systems
        font_paths = [
            "/System/Library/Fonts/Supplemental/Impact.ttf",  # macOS
            "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",  # Ubuntu/Debian
            "C:\\Windows\\Fonts\\impact.ttf",  # Windows
            "/usr/share/fonts/TTF/impact.ttf",  # Arch Linux
        ]

        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, size)
                    break  # Found it!
                except IOError:
                    continue  # Try next path

        # Fallback if Impact not found
        if font is None:
            try:
                font = ImageFont.truetype("arial.ttf", size)
            except IOError:
                # Last resort: PIL's built-in default font (looks bad but works)
                font = ImageFont.load_default()
                print("⚠️  Could not load Impact font, using default")

        # Cache the font for future use
        self.font_cache[size] = font
        return font

    def draw_outlined_text(
        self,
        draw: ImageDraw.ImageDraw,
        position: Tuple[int, int],
        text: str,
        font: ImageFont.FreeTypeFont,
        fill_color: str = "white",
        outline_color: str = "black",
        outline_width: int = 3,
    ):
        """
        Draw text with an outline (classic meme style).

        Memes use white text with a black outline so the text is readable
        on any background color. We achieve this by:
        1. Drawing the text multiple times in black, slightly offset in all directions
        2. Drawing the text once in white on top

        Args:
            draw: PIL ImageDraw object (canvas to draw on)
            position: (x, y) coordinates for text center
            text: The text to draw
            font: Font to use
            fill_color: Main text color (default: white)
            outline_color: Outline color (default: black)
            outline_width: How thick the outline should be in pixels
        """
        x, y = position

        # Draw the outline by drawing the text multiple times, offset in all directions
        # This creates a "shadow" effect around the text
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                # Skip the center position (that's where the main text goes)
                if dx != 0 or dy != 0:
                    draw.text(
                        (x + dx, y + dy),  # Offset position
                        text,
                        font=font,
                        fill=outline_color,
                        anchor="mm"  # "mm" = middle-middle (center the text on the point)
                    )

        # Draw the main (white) text on top
        draw.text((x, y), text, font=font, fill=fill_color, anchor="mm")

    def wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        """
        Wrap text to fit within a maximum width.

        If the text is too long for one line, this splits it into multiple lines.
        We can't use Python's textwrap module because we need to measure
        actual pixel width with the specific font.

        Args:
            text: The text to wrap
            font: Font being used (needed to measure text width)
            max_width: Maximum width in pixels

        Returns:
            List of text lines that fit within max_width
        """
        words = text.split()
        lines = []
        current_line = []  # Words for the current line

        for word in words:
            # Try adding this word to the current line
            test_line = " ".join(current_line + [word])

            # Measure how wide this line would be
            bbox = font.getbbox(test_line)  # Returns (left, top, right, bottom)
            width = bbox[2] - bbox[0]       # right - left = width

            if width <= max_width:
                # It fits! Add the word to current line
                current_line.append(word)
            else:
                # Too wide! Start a new line
                if current_line:  # Save current line if not empty
                    lines.append(" ".join(current_line))
                current_line = [word]  # Start new line with this word

        # Don't forget the last line
        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def add_text_to_image(
        self,
        image: Image.Image,
        text_data: Dict[str, str],
        template_key: str,
    ) -> Image.Image:
        """
        Add meme text to an image using the correct positioning.

        This is where we actually draw the text onto the meme template.
        It handles text wrapping, positioning, and the outline effect.

        Args:
            image: The meme template image
            text_data: Dict with "top", "bottom", "middle" text
            template_key: Name of template (to look up positioning)

        Returns:
            New image with text added (original is not modified)
        """
        # Make a copy so we don't modify the original
        img = image.copy()

        # Create a drawing context
        draw = ImageDraw.Draw(img)

        width, height = img.size

        # Get text positions for this template (or use default)
        positions = MEME_TEXT_POSITIONS.get(
            template_key,
            MEME_TEXT_POSITIONS["default"]
        )

        # Calculate font size based on image dimensions
        # Smaller dimension / 12 gives a reasonable meme-sized font
        base_font_size = min(width, height) // 12
        font = self.get_font(base_font_size)

        # Draw each text field (top, middle, bottom, caption)
        for key in ["top", "middle", "bottom", "caption"]:
            text = text_data.get(key, "").strip()

            # Skip empty text or text positions not defined for this template
            if not text or key not in positions:
                continue

            # Get position config for this text field
            pos_config = positions[key]

            # Convert ratio positions to actual pixel coordinates
            x = int(width * pos_config["pos"][0])
            y = int(height * pos_config["pos"][1])
            max_text_width = int(width * pos_config["max_width"])

            # Wrap text to fit within max width
            # Also convert to UPPERCASE (classic meme style!)
            lines = self.wrap_text(text.upper(), font, max_text_width)

            # Calculate total height of the text block
            line_height = base_font_size + 5  # Add a bit of spacing
            total_height = len(lines) * line_height

            # Adjust Y so the text block is centered on the target position
            start_y = y - total_height // 2

            # Draw each line
            for i, line in enumerate(lines):
                line_y = start_y + i * line_height + line_height // 2
                self.draw_outlined_text(draw, (x, line_y), line, font)

        return img

    def get_template_image(self, meme_name: str) -> Tuple[Optional[Image.Image], str]:
        """
        Load a meme template image from disk.

        Args:
            meme_name: Name of the meme (e.g., "Drake")

        Returns:
            Tuple of (loaded image, template_key) or (None, "") if not found
        """
        # Look up the filename for this meme name
        template_file = MEME_TEMPLATES.get(meme_name)

        # If exact match not found, try fuzzy matching
        if template_file is None:
            meme_name_lower = meme_name.lower()
            for name, file in MEME_TEMPLATES.items():
                # Check if names partially match
                if meme_name_lower in name.lower() or name.lower() in meme_name_lower:
                    template_file = file
                    break

        if template_file is None:
            print(f"⚠️  No template found for '{meme_name}'")
            return None, ""

        template_path = TEMPLATES_DIR / template_file

        if not template_path.exists():
            print(f"⚠️  Template file not found: {template_path}")
            print("   Run download_memes.py to download templates")
            return None, ""

        # Get the template key (filename without extension) for position lookup
        template_key = template_file.replace(".jpg", "").replace(".png", "")

        # Load and return the image
        return Image.open(template_path), template_key

    def generate_meme(
        self,
        meme_name: str = "Drake",
        custom_text: Optional[Dict[str, str]] = None,
        temperature: float = 0.8,
        output_name: Optional[str] = None,
        use_openai: bool = False,
        topic: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
    ) -> Optional[Path]:
        """
        Generate a complete meme image with text overlay.

        This is the main method that ties everything together:
        1. Gets the meme text (custom, OpenAI, or local model)
        2. Loads the template image
        3. Draws the text on the image
        4. Saves the result

        Args:
            meme_name: Which meme format to use (e.g., "Drake")
            custom_text: Your own text as {"top": "...", "bottom": "..."}
            temperature: Creativity for local model (0.0-1.0+)
            output_name: Custom filename for the output (auto-generated if not provided)
            use_openai: Use OpenAI API instead of local model
            topic: Topic hint for AI generation (e.g., "programming")
            openai_model: Which OpenAI model to use

        Returns:
            Path to the saved meme image, or None if generation failed
        """
        print(f"\n🎨 Generating {meme_name} meme...")

        # Step 1: Load the template image
        template_img, template_key = self.get_template_image(meme_name)

        if template_img is None:
            return None

        # Step 2: Get the text (from user, OpenAI, or local model)
        if custom_text:
            # User provided their own text
            text_data = {"meme_name": meme_name, **custom_text}
            print(f"   Using custom text")
        elif use_openai:
            # Use OpenAI API (high quality)
            text_data = self.generate_meme_text_openai(meme_name, topic, openai_model)
            print(f"   Generated (OpenAI): {text_data}")
        else:
            # Use local GPT-2 model (lower quality but free)
            generated = self.generate_meme_text(meme_name, temperature)
            text_data = self.parse_meme_text(generated)
            print(f"   Generated (local): {generated}")

        # Step 3: Draw text on the image
        final_img = self.add_text_to_image(template_img, text_data, template_key)

        # Step 4: Save the result
        if output_name is None:
            # Generate a unique filename using timestamp
            import time
            timestamp = int(time.time())
            output_name = f"{meme_name.lower().replace(' ', '_')}_{timestamp}.jpg"

        output_path = OUTPUT_DIR / output_name
        final_img.save(output_path, quality=95)  # High quality JPEG

        print(f"✅ Saved to: {output_path}")
        return output_path

    def generate_batch(
        self,
        meme_names: List[str],
        count_per_meme: int = 3,
        temperature: float = 0.8,
    ) -> List[Path]:
        """
        Generate multiple memes at once.

        Args:
            meme_names: List of meme formats to use
            count_per_meme: How many variations of each
            temperature: Creativity level for local model

        Returns:
            List of paths to generated meme images
        """
        results = []

        for meme_name in meme_names:
            print(f"\n{'='*50}")
            print(f"Generating {count_per_meme} variations of '{meme_name}'")

            for i in range(count_per_meme):
                result = self.generate_meme(meme_name, temperature=temperature)
                if result:
                    results.append(result)

        return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================
# This code runs when you execute the script directly:
#   python generate_memes.py --meme "Drake" --top "Hello" --bottom "World"

def main():
    """Main entry point for command line usage."""
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate memes with AI-generated text")
    parser.add_argument("--meme", type=str, default="Drake", help="Meme format name")
    parser.add_argument("--top", type=str, help="Custom top text")
    parser.add_argument("--bottom", type=str, help="Custom bottom text")
    parser.add_argument("--batch", action="store_true", help="Generate batch of memes")
    parser.add_argument("--count", type=int, default=3, help="Number of memes per format in batch mode")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--list", action="store_true", help="List available meme formats")

    args = parser.parse_args()

    # Handle --list flag: show available memes
    if args.list:
        print("📋 Available Meme Formats:")
        print("-" * 40)
        for name, file in sorted(MEME_TEMPLATES.items()):
            template_exists = (TEMPLATES_DIR / file).exists()
            status = "✅" if template_exists else "❌ (not downloaded)"
            print(f"  {status} {name}")
        return

    # Create the generator
    generator = MemeGenerator()

    if args.batch:
        # Batch mode: generate multiple memes
        main_memes = ["Drake", "Distracted Boyfriend", "One Does Not Simply"]
        results = generator.generate_batch(main_memes, args.count, args.temperature)

        print(f"\n{'='*50}")
        print(f"🎉 Generated {len(results)} memes!")
        print(f"📁 Output directory: {OUTPUT_DIR}")
    else:
        # Single meme mode
        custom_text = None
        if args.top or args.bottom:
            custom_text = {}
            if args.top:
                custom_text["top"] = args.top
            if args.bottom:
                custom_text["bottom"] = args.bottom

        generator.generate_meme(
            meme_name=args.meme,
            custom_text=custom_text,
            temperature=args.temperature,
        )


# This runs main() only when the script is executed directly,
# not when it's imported as a module
if __name__ == "__main__":
    main()
