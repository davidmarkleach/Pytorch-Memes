#!/usr/bin/env python3
"""
Streamlit Meme Generator UI
Run with: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
from PIL import Image
import io
import time
import os
import random

from generate_memes import MemeGenerator, MEME_TEMPLATES, TEMPLATES_DIR, OPENAI_AVAILABLE

# Page config
st.set_page_config(
    page_title="🎭 PyTorch Meme Generator",
    page_icon="🎭",
    layout="wide"
)

# Custom CSS for meme aesthetic
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    h1, h2, h3 {
        color: #e94560 !important;
    }
    h1 {
        text-align: center;
        font-family: 'Impact', sans-serif;
        text-shadow: 2px 2px 4px #000;
    }
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #ff6b6b);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff6b6b, #e94560);
        transform: scale(1.02);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
    }
    .meme-card {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    /* Thumbnail selector styling */
    div[data-testid="stSidebar"] img {
        border-radius: 8px;
        border: 2px solid transparent;
        transition: all 0.2s ease;
    }
    div[data-testid="stSidebar"] img:hover {
        border-color: #e94560;
        transform: scale(1.05);
    }
    /* Compact buttons for template selector */
    div[data-testid="stSidebar"] .stButton button {
        font-size: 0.75rem;
        padding: 0.25rem 0.5rem;
        margin-top: -5px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize generator (cached)
@st.cache_resource
def get_generator(api_key: str = None):
    gen = MemeGenerator(openai_api_key=api_key)
    return gen


# Get available templates
def get_available_templates():
    available = []
    for name, filename in MEME_TEMPLATES.items():
        if (TEMPLATES_DIR / filename).exists():
            available.append(name)
    return available


# Title
st.title("🎭 PyTorch Meme Generator")
st.markdown("---")

# Sidebar for options
with st.sidebar:
    st.header("⚙️ Settings")

    # OpenAI API Key
    st.subheader("🤖 AI Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key for AI-generated memes",
        placeholder="sk-..."
    )

    if api_key:
        st.success("✅ API key set!")
        os.environ["OPENAI_API_KEY"] = api_key

    available_templates = get_available_templates()

    if not available_templates:
        st.error("No templates found! Run `python download_memes.py` first.")
        st.stop()

    # OpenAI model selection
    if api_key:
        openai_model = st.selectbox(
            "OpenAI Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            help="gpt-4o-mini is fast & cheap, gpt-4o is smartest"
        )
    else:
        openai_model = "gpt-4o-mini"

    st.markdown("---")

    # === BATCH MODE TOGGLE ===
    st.subheader("🎲 Generation Mode")

    batch_mode = st.toggle("**Batch Mode**", value=False, help="Generate multiple memes with one topic")

    if batch_mode:
        num_memes = st.slider(
            "Number of memes",
            min_value=1,
            max_value=min(10, len(available_templates)),
            value=3,
            help="How many different meme templates to use"
        )
        st.info(f"Will generate {num_memes} random meme formats")
    else:
        num_memes = 1
        # Template selection (only in single mode)
        st.markdown("---")
        st.subheader("🖼️ Template")

        # Initialize selected meme in session state
        if "selected_meme" not in st.session_state:
            st.session_state.selected_meme = available_templates[0] if available_templates else "Drake"

        # Visual template selector with thumbnails
        st.markdown("**Click to select:**")

        # Create a scrollable container with thumbnail grid
        cols_per_row = 3
        for row_start in range(0, min(12, len(available_templates)), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                template_idx = row_start + col_idx
                if template_idx < len(available_templates):
                    meme_name = available_templates[template_idx]
                    template_file = MEME_TEMPLATES.get(meme_name)

                    with cols[col_idx]:
                        if template_file:
                            template_path = TEMPLATES_DIR / template_file
                            if template_path.exists():
                                # Show small thumbnail
                                st.image(
                                    str(template_path),
                                    width=80,
                                    caption=None
                                )

                        # Selection button
                        is_selected = st.session_state.selected_meme == meme_name
                        button_label = f"{'✅ ' if is_selected else ''}{meme_name[:12]}{'...' if len(meme_name) > 12 else ''}"

                        if st.button(
                            button_label,
                            key=f"select_{meme_name}",
                            use_container_width=True,
                            type="primary" if is_selected else "secondary"
                        ):
                            st.session_state.selected_meme = meme_name
                            # No st.rerun() needed - button click already triggers rerun

        # Show more templates in expander if there are many
        if len(available_templates) > 12:
            with st.expander(f"Show all {len(available_templates)} templates"):
                expander_selection = st.selectbox(
                    "All templates",
                    available_templates,
                    index=available_templates.index(st.session_state.selected_meme) if st.session_state.selected_meme in available_templates else 0,
                    key="all_templates_select"
                )
                if expander_selection != st.session_state.selected_meme:
                    st.session_state.selected_meme = expander_selection
                    # No st.rerun() needed - selectbox change already triggers rerun

        # Use session state for selected meme
        selected_meme = st.session_state.selected_meme

        st.success(f"Selected: **{selected_meme}**")

    st.markdown("---")
    st.markdown("### 📋 Quick Tips")
    st.markdown("""
    - **Batch mode**: Enter a topic, get multiple memes!
    - **Single mode**: Pick a specific template
    - **Topics**: "programming", "coffee", "mondays"
    """)

# Main content area
if batch_mode:
    # === BATCH MODE UI ===
    st.header("🎲 Batch Meme Generator")
    st.markdown("Enter a topic and generate multiple memes with random templates!")

    col1, col2 = st.columns([2, 1])

    with col1:
        topic = st.text_input(
            "🎯 Topic",
            placeholder="e.g., programming, coffee addiction, working from home, gym...",
            help="What should the memes be about?"
        )

        if not api_key:
            st.warning("⚠️ Enter OpenAI API key in sidebar for best results!")

        generate_clicked = st.button("🎨 Generate Memes!", use_container_width=True)

    with col2:
        st.metric("Templates Available", len(available_templates))
        st.metric("Memes to Generate", num_memes)

    # Generate batch
    if generate_clicked:
        if not topic:
            st.warning("Please enter a topic!")
            st.stop()

        if not api_key:
            st.error("OpenAI API key required for batch mode!")
            st.stop()

        # Pick random templates
        templates_to_use = random.sample(available_templates, num_memes)

        st.markdown("---")
        st.header(f"🎉 Generating {num_memes} Memes about '{topic}'...")

        generator = get_generator(api_key)
        generator.set_openai_key(api_key)

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        generated_memes = []

        for i, meme_name in enumerate(templates_to_use):
            status_text.text(f"Creating {meme_name} meme... ({i+1}/{num_memes})")

            try:
                result_path = generator.generate_meme(
                    meme_name=meme_name,
                    use_openai=True,
                    topic=topic,
                    openai_model=openai_model
                )

                if result_path and result_path.exists():
                    generated_memes.append((meme_name, result_path))
            except Exception as e:
                st.warning(f"Failed to generate {meme_name}: {e}")

            progress_bar.progress((i + 1) / num_memes)

        status_text.text("Done!")
        progress_bar.empty()

        # Display results in grid
        if generated_memes:
            st.success(f"✅ Generated {len(generated_memes)} memes!")

            # Create columns for grid display
            cols_per_row = 3

            for row_start in range(0, len(generated_memes), cols_per_row):
                cols = st.columns(cols_per_row)

                for col_idx, (meme_name, path) in enumerate(generated_memes[row_start:row_start + cols_per_row]):
                    with cols[col_idx]:
                        img = Image.open(path)
                        st.image(img, caption=meme_name, use_container_width=True)

                        # Download button for each
                        img_bytes = io.BytesIO()
                        img.save(img_bytes, format='JPEG', quality=95)
                        img_bytes.seek(0)

                        st.download_button(
                            label="📥 Download",
                            data=img_bytes,
                            file_name=f"{meme_name.lower().replace(' ', '_')}_{int(time.time())}.jpg",
                            mime="image/jpeg",
                            use_container_width=True,
                            key=f"dl_{meme_name}_{row_start}_{col_idx}"
                        )

else:
    # === SINGLE MODE UI ===
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Create Your Meme")

        # Mode selection
        mode = st.radio(
            "Text Source",
            ["Custom Text", "AI Generated (OpenAI)"] if api_key else ["Custom Text", "AI Generated (Local)"],
            horizontal=True,
            key="text_source_mode"
        )

        if "Custom" in mode:
            top_text = st.text_area("Top Text", placeholder="Enter top text...", height=80)
            bottom_text = st.text_area("Bottom Text", placeholder="Enter bottom text...", height=80)

            if selected_meme in ["Distracted Boyfriend", "Expanding Brain", "Gru's Plan"]:
                middle_text = st.text_area("Middle Text (optional)", placeholder="Enter middle text...", height=80)
            else:
                middle_text = ""
            topic = ""
        else:
            topic = st.text_input(
                "Topic (optional)",
                placeholder="e.g., programming, coffee, mondays...",
                help="Give the AI a topic hint"
            )
            top_text = ""
            bottom_text = ""
            middle_text = ""

            if "OpenAI" in mode:
                st.success("🚀 Using OpenAI!")
            else:
                st.warning("⚠️ Local model - add API key for better results")

        generate_clicked = st.button("🎨 Generate Meme!", use_container_width=True)

    with col2:
        st.header("🖼️ Template Preview")
        template_file = MEME_TEMPLATES.get(selected_meme)
        if template_file:
            template_path = TEMPLATES_DIR / template_file
            if template_path.exists():
                st.image(str(template_path), caption=f"Template: {selected_meme}", use_container_width=True)

    # Generate single meme
    if generate_clicked:
        with st.spinner("🎭 Generating your meme..."):
            try:
                generator = get_generator(api_key if api_key else None)
                if api_key:
                    generator.set_openai_key(api_key)

                if "Custom" in mode:
                    if not top_text and not bottom_text:
                        st.warning("Please enter some text!")
                        st.stop()

                    custom_text = {"top": top_text, "bottom": bottom_text}
                    if middle_text:
                        custom_text["middle"] = middle_text

                    result_path = generator.generate_meme(
                        meme_name=selected_meme,
                        custom_text=custom_text
                    )
                elif "OpenAI" in mode:
                    result_path = generator.generate_meme(
                        meme_name=selected_meme,
                        use_openai=True,
                        topic=topic if topic else None,
                        openai_model=openai_model
                    )
                else:
                    result_path = generator.generate_meme(
                        meme_name=selected_meme,
                        temperature=0.8
                    )

                if result_path and result_path.exists():
                    st.success("✅ Meme generated!")
                    st.markdown("---")
                    st.header("🎉 Your Meme")

                    result_img = Image.open(result_path)
                    st.image(result_img, caption="Your generated meme!", use_container_width=True)

                    img_bytes = io.BytesIO()
                    result_img.save(img_bytes, format='JPEG', quality=95)
                    img_bytes.seek(0)

                    st.download_button(
                        label="📥 Download Meme",
                        data=img_bytes,
                        file_name=f"meme_{selected_meme.lower().replace(' ', '_')}_{int(time.time())}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
                else:
                    st.error("Failed to generate meme.")

            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>🎭 Made with PyTorch & OpenAI</p>",
    unsafe_allow_html=True
)
