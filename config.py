# config.py
import json
import os

DEFAULT_CONFIG_FILENAME = "config.json"
DEFAULT_PROMPTS = {
    "image_analysis": "Analyze this image. Suggest a detailed and descriptive filename base (lowercase, underscores for spaces, no file extension). If the image features a recurring subject or scene element, try to use a consistent and specific term for that element. Example for a general scene: 'ginger_cat_sleeping_blue_armchair_sunlit_room'. Avoid generic names.\n\n{metadata_prompt_section}Only output the suggested filename base.",
    "video_frame_analysis": "This is a frame from a video. Describe this frame in detail. Focus on key subjects, actions, and setting. If a clear, recurring subject is visible, mention it consistently.\n\n{metadata_prompt_section}Only output the description of this single frame.",
    "video_summary": "The following are descriptions of sequential scenes from a video. Synthesize these descriptions to understand the video's overall theme or main action. Suggest a detailed filename base that captures this theme. **Crucially, do NOT use the word 'frame' or 'scene' in the filename.** If the descriptions are vague or describe poor quality (e.g., 'blurry'), name it based on that, like 'blurry_video_of_cat'. Avoid single-word or single-number names.\n\n{metadata_prompt_section}Sequential Scene Descriptions:\n{combined_descriptions}\n\nOnly output the suggested filename base.",
    "pdf_analysis": "Analyze the text content from a PDF below. Suggest a short, descriptive filename base (lowercase, underscores for spaces). DO NOT include file extensions. DO NOT explain your reasoning. Your entire response must be the filename base ONLY. Example: 'q3_marketing_notes'.\n\n{metadata_prompt_section}PDF Text content:\n{content_to_send_for_llm}\n\nFilename base:",
    "text_analysis": "Analyze the text content below. Suggest a short, descriptive filename base (lowercase, underscores for spaces). DO NOT include file extensions. DO NOT explain your reasoning. Your entire response must be the filename base ONLY. Example: 'web_scraper_for_news_articles'.\n\n{metadata_prompt_section}Text content:\n{content_to_send_for_llm}\n\nFilename base:",
    "filename_grouping": "Your task is to group the filenames in the following JSON list. Group files that refer to the same subject. For each group, create a single, uniform base name. Respond ONLY with a valid JSON object where each key is the new uniform base name, and the value is a list of the original filenames in that group.\n\nExample Input:\n[\"cat_on_sofa.jpg\", \"dog_in_park.jpg\", \"sofa_with_cat.jpg\"]\n\nExample Output:\n{\n  \"cat_on_sofa\": [\"cat_on_sofa.jpg\", \"sofa_with_cat.jpg\"],\n  \"dog_in_park\": [\"dog_in_park.jpg\"]\n}\n\nStrictly adhere to the JSON output format. Do not add any other text.\n\nProcess this list:\n{filename_list_json}"
}
DEFAULT_CONFIG_DATA = {
  "models": {"text_model": "gemma:2b", "vision_model": "llava:latest"},
  "ollama_host": "http://localhost:11434",
  "generation_parameters": {"temperature": 0.4, "num_predict": 70, "max_text_chars": 2000},
  "video_processing": {"frames_to_analyze": 5},
  "prompts": DEFAULT_PROMPTS
}

def load_config(filename=DEFAULT_CONFIG_FILENAME):
    """Loads configuration from a JSON file, using defaults for missing keys."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            # Ensure all default top-level keys exist
            for key, value in DEFAULT_CONFIG_DATA.items():
                config.setdefault(key, value)
            # Ensure all default prompt keys exist
            if "prompts" in config:
                for key, value in DEFAULT_PROMPTS.items():
                    config["prompts"].setdefault(key, value)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading {filename}: {e}. Using default config.")
            config = DEFAULT_CONFIG_DATA.copy()
    else:
        print(f"Config file {filename} not found. Creating with default values.")
        config = DEFAULT_CONFIG_DATA.copy()
        save_config(config, filename)
    return config

def save_config(config, filename=DEFAULT_CONFIG_FILENAME):
    """Saves the configuration to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved configuration to {filename}")
    except Exception as e:
        print(f"Error saving config to {filename}: {e}")