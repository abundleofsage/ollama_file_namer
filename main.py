#!/usr/bin/env python3
import ollama
import os
import argparse
import base64
import mimetypes
import re
import sys
import datetime # Added for formatting timestamps
import subprocess # For calling FFmpeg
import tempfile # For temporary frame files
import json # For parsing config and ffprobe output

# Attempt to import pypdf for PDF text extraction
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    PdfReader = None # To avoid NameError if not available

# Global Ollama client
CLIENT = None
# Global config dictionary
CONFIG = {}

# --- Configuration Handling ---
DEFAULT_CONFIG_FILENAME = "config.json"

DEFAULT_PROMPTS = {
    "image_analysis": "Analyze this image. Suggest a detailed and descriptive filename base (lowercase, underscores for spaces, no file extension). If the image features a recurring subject or scene element, try to use a consistent and specific term for that element. Example for a general scene: 'ginger_cat_sleeping_blue_armchair_sunlit_room'. Avoid generic names.\n\n{metadata_prompt_section}Only output the suggested filename base.",
    "video_frame_analysis": "This is frame {frame_num} of {total_frames} from a video. Describe this video frame in detail. Focus on key subjects, actions, and setting. If a clear, recurring subject is visible, mention it consistently.\n\n{metadata_prompt_section}Only output the description of this single frame.",
    "video_summary": "The following are sequential descriptions of scenes from a video. Based *only* on the content of these descriptions, suggest a detailed and descriptive filename base (lowercase, underscores for spaces, no file extension). Focus on the main theme, subject, or overall progression depicted in these scenes. If a recurring subject or theme is evident, ensure this recurring element is part of the filename base. Do NOT include literal text like 'frame 1' in the filename itself. Avoid generic names.\n\n{metadata_prompt_section}Sequential Scene Descriptions:\n{combined_descriptions}\n\nOnly output the suggested filename base.",
    "pdf_analysis": "Analyze the following text content (extracted from a PDF). Suggest a detailed and descriptive filename base (lowercase, underscores for spaces, no file extension). Example: 'project_phoenix_q3_marketing_strategy_meeting_notes'. Avoid generic names.\n\n{metadata_prompt_section}PDF Text content (first {max_text_chars} chars, if available):\n{content_to_send_for_llm}\n\nOnly output the suggested filename base.",
    "text_analysis": "Analyze the following text content. Suggest a detailed and descriptive filename base (lowercase, underscores for spaces, no file extension). Example: 'project_phoenix_q3_marketing_strategy_meeting_notes'. Avoid generic names.\n\n{metadata_prompt_section}Text content (first {max_text_chars} chars, if available):\n{content_to_send_for_llm}\n\nOnly output the suggested filename base."
}

DEFAULT_CONFIG_DATA = {
  "models": {
    "text_model": "gemma:2b",
    "vision_model": "llava:latest"
  },
  "ollama_host": "http://localhost:11434",
  "generation_parameters": {
    "temperature": 0.4,
    "num_predict": 70
  },
  "video_processing": {
    "frames_to_analyze": 5
  },
  "prompts": DEFAULT_PROMPTS
}

def load_config(filename=DEFAULT_CONFIG_FILENAME):
    """Loads configuration from a JSON file."""
    global CONFIG
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                CONFIG = json.load(f)
            print(f"Loaded configuration from {filename}")
            # Ensure all default prompt keys exist, add them if not (for backward compatibility)
            if "prompts" not in CONFIG:
                CONFIG["prompts"] = {}
            for key, default_prompt_value in DEFAULT_PROMPTS.items():
                CONFIG["prompts"].setdefault(key, default_prompt_value)

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {filename}. Using default configuration.")
            CONFIG = DEFAULT_CONFIG_DATA.copy() # Use a copy
        except Exception as e:
            print(f"Error loading config file {filename}: {e}. Using default configuration.")
            CONFIG = DEFAULT_CONFIG_DATA.copy()
    else:
        print(f"Configuration file {filename} not found. Creating with default values.")
        CONFIG = DEFAULT_CONFIG_DATA.copy()
        save_config(filename) # Save the default config

def save_config(filename=DEFAULT_CONFIG_FILENAME):
    """Saves the current configuration to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(CONFIG, f, indent=2)
        print(f"Saved configuration to {filename}")
    except Exception as e:
        print(f"Error saving configuration to {filename}: {e}")

# --- End Configuration Handling ---


def sanitize_filename_component(component):
    name = component.strip().strip("'").strip('"')
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^\w._-]', '', name) 
    return name.lower()

def get_file_metadata(filepath, skip_metadata=False):
    if skip_metadata:
        return "" 

    try:
        stat_info = os.stat(filepath)
        creation_time = datetime.datetime.fromtimestamp(stat_info.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        modification_time = datetime.datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        size_kb = round(stat_info.st_size / 1024, 2)
        
        metadata_str = (
            f"Original filename: '{os.path.basename(filepath)}'\n"
            f"Creation time: {creation_time}\n"
            f"Modification time: {modification_time}\n"
            f"File size: {size_kb} KB"
        )
        current_file_mime_type, _ = mimetypes.guess_type(filepath)
        original_ext_lower = os.path.splitext(filepath)[1].lower()
        
        known_video_extensions_for_metadata = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.mpeg', '.mpg', '.mts']
        is_known_video_ext = original_ext_lower in known_video_extensions_for_metadata

        if (current_file_mime_type and current_file_mime_type.startswith('video/')) or is_known_video_ext:
            try:
                duration_str = get_video_duration_str(filepath)
                if duration_str:
                    metadata_str += f"\nDuration: {duration_str}"
            except Exception as e:
                print(f"  Could not get video duration for {filepath}: {e}")
        
        return metadata_str
    except Exception as e:
        print(f"  Could not retrieve metadata for {filepath}: {e}")
        return f"Original filename: '{os.path.basename(filepath)}'\n[Could not retrieve full metadata]"


def get_video_duration_str(video_filepath):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_filepath]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration_seconds = float(result.stdout.strip())
        td = datetime.timedelta(seconds=duration_seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_formatted = ""
        if td.days > 0: duration_formatted += f"{td.days}d "
        if hours > 0: duration_formatted += f"{hours}h "
        if minutes > 0: duration_formatted += f"{minutes}m "
        duration_formatted += f"{seconds}s"
        return duration_formatted.strip()
    except subprocess.CalledProcessError as e:
        print(f"  ffprobe error getting duration for {video_filepath}: {e.stderr}")
    except FileNotFoundError:
        print("  Error: ffprobe command not found. Is FFmpeg installed and in PATH?")
    except Exception as e:
        print(f"  Unexpected error getting video duration for {video_filepath}: {e}")
    return None


def extract_frames_as_base64(video_filepath, num_frames_to_extract, temp_dir):
    base64_frames = []
    temp_frame_files = []
    try:
        cmd_duration = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_filepath]
        result_duration = subprocess.run(cmd_duration, capture_output=True, text=True, check=True)
        duration = float(result_duration.stdout.strip())

        if duration <= 0:
            print(f"  Video {video_filepath} has zero or negative duration. Skipping frame extraction.")
            return []
        if num_frames_to_extract <= 0: num_frames_to_extract = 1 
        interval = duration / (num_frames_to_extract + 1) 
        timestamps = [interval * (i + 1) for i in range(num_frames_to_extract)]
        if not timestamps and duration > 0 : timestamps = [duration / 2] 
        elif not timestamps and duration <=0 : 
             print(f"  Video {video_filepath} has zero or negative duration after check. Skipping frame extraction.")
             return []

        print(f"  Extracting {len(timestamps)} frames from {os.path.basename(video_filepath)} (duration: {duration:.2f}s)...")
        for i, ts in enumerate(timestamps):
            temp_frame_path = os.path.join(temp_dir, f"frame_{i+1}_{os.path.basename(video_filepath)}.png")
            temp_frame_files.append(temp_frame_path)
            cmd_extract = ['ffmpeg', '-ss', str(ts), '-i', video_filepath, '-frames:v', '1', '-q:v', '2', '-y', temp_frame_path]
            process_extract = subprocess.run(cmd_extract, capture_output=True, text=True)
            if process_extract.returncode != 0:
                print(f"  FFmpeg error extracting frame for {video_filepath} at {ts:.2f}s: {process_extract.stderr}")
                continue 
            if os.path.exists(temp_frame_path) and os.path.getsize(temp_frame_path) > 0:
                with open(temp_frame_path, 'rb') as image_file:
                    img_b64 = base64.b64encode(image_file.read()).decode('utf-8')
                    base64_frames.append(img_b64)
            else:
                print(f"  Warning: Could not extract or read frame at timestamp {ts:.2f}s for {video_filepath}")
    except subprocess.CalledProcessError as e:
        print(f"  FFmpeg/ffprobe error during frame extraction for {video_filepath}: {e.stderr}")
    except FileNotFoundError:
        print("  Error: ffmpeg or ffprobe command not found. Is FFmpeg installed and in PATH?")
    except Exception as e:
        print(f"  Unexpected error during frame extraction for {video_filepath}: {e}")
    finally:
        for temp_file in temp_frame_files:
            if os.path.exists(temp_file):
                try: os.remove(temp_file)
                except Exception as e_del: print(f"  Warning: Could not delete temporary frame file {temp_file}: {e_del}")
    return base64_frames


def get_suggested_name(filepath, file_type, text_model_cfg, vision_model_cfg, ollama_client,
                       max_text_chars_cfg, temperature_cfg, num_predict_cfg,
                       video_frames_to_analyze_cfg, temp_dir_for_frames,
                       skip_metadata_flag, prompts_cfg):
    original_full_filename = os.path.basename(filepath)
    original_basename, original_ext = os.path.splitext(original_full_filename)
    original_ext = original_ext.lower()

    file_metadata_info = get_file_metadata(filepath, skip_metadata_flag)
    ollama_options = {"temperature": temperature_cfg, "num_predict": num_predict_cfg}
    suggested_base = ""
    content_to_send_for_llm = "" 

    metadata_prompt_section = ""
    if file_metadata_info:
        metadata_prompt_section = f"File Metadata:\n{file_metadata_info}\n\n"

    try:
        if file_type and file_type.startswith('image/'): 
            if not vision_model_cfg:
                print(f"Vision model not specified, skipping image {original_full_filename}")
                return None
            with open(filepath, 'rb') as image_file:
                img_b64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt_template = prompts_cfg.get("image_analysis", DEFAULT_PROMPTS["image_analysis"])
            prompt = prompt_template.format(metadata_prompt_section=metadata_prompt_section)
            
            print(f"  Analyzing image with {vision_model_cfg} (Temp: {temperature_cfg}, Predict: {num_predict_cfg})...")
            response = ollama_client.generate(model=vision_model_cfg, prompt=prompt, images=[img_b64], stream=False, options=ollama_options)
            suggested_base = response['response'].strip()

        elif file_type and file_type.startswith('video/'): 
            if not vision_model_cfg or not text_model_cfg:
                print(f"  Vision or Text model not specified, skipping video {original_full_filename}")
                return None
            if not temp_dir_for_frames:
                print(f"  Temporary directory for frames not provided. Skipping video {original_full_filename}")
                return None

            print(f"  Processing video: {original_full_filename}")
            b64_frames = extract_frames_as_base64(filepath, video_frames_to_analyze_cfg, temp_dir_for_frames)

            if not b64_frames:
                print(f"  No frames extracted or analyzed for video {original_full_filename}. Skipping.")
                return None

            frame_descriptions_content_only = [] 
            print(f"  Analyzing {len(b64_frames)} frames with {vision_model_cfg}...")
            prompt_frame_template = prompts_cfg.get("video_frame_analysis", DEFAULT_PROMPTS["video_frame_analysis"])
            for i, frame_b64 in enumerate(b64_frames):
                prompt_frame = prompt_frame_template.format(
                    frame_num=i+1, 
                    total_frames=len(b64_frames), 
                    metadata_prompt_section=metadata_prompt_section
                )
                try:
                    response_frame = ollama_client.generate(model=vision_model_cfg, prompt=prompt_frame, images=[frame_b64], stream=False, options=ollama_options)
                    description = response_frame['response'].strip()
                    if description: frame_descriptions_content_only.append(description) 
                    print(f"    Frame {i+1} description: {description[:100]}...") 
                except Exception as e_frame:
                    print(f"    Error analyzing frame {i+1} for {original_full_filename}: {e_frame}")
            
            if not frame_descriptions_content_only:
                print(f"  No valid descriptions obtained from frames for {original_full_filename}. Skipping.")
                return None

            combined_descriptions_for_summary = "\n".join(frame_descriptions_content_only)
            prompt_summary_template = prompts_cfg.get("video_summary", DEFAULT_PROMPTS["video_summary"])
            prompt_summary = prompt_summary_template.format(
                metadata_prompt_section=metadata_prompt_section,
                combined_descriptions=combined_descriptions_for_summary
            )
            print(f"  Summarizing frame descriptions with {text_model_cfg}...")
            response_summary = ollama_client.generate(model=text_model_cfg, prompt=prompt_summary, stream=False, options=ollama_options)
            suggested_base = response_summary['response'].strip()
        
        elif file_type and file_type == 'application/pdf':
            if not text_model_cfg:
                print(f"Text model not specified, skipping PDF file {original_full_filename}")
                return None
            print(f"  Processing PDF: {original_full_filename}")
            if PYPDF_AVAILABLE and PdfReader:
                try:
                    reader = PdfReader(filepath)
                    extracted_text = ""
                    num_pages_to_read = min(len(reader.pages), 50) 
                    for page_num in range(num_pages_to_read):
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text: extracted_text += page_text + "\n" 
                        if len(extracted_text) >= max_text_chars_cfg: break
                    if extracted_text.strip():
                        content_to_send_for_llm = extracted_text[:max_text_chars_cfg]
                        print(f"  Successfully extracted ~{len(content_to_send_for_llm)} chars of text from PDF (up to {num_pages_to_read} pages).")
                    else:
                        print(f"  Could not extract text from PDF (it might be image-based, empty, or encrypted).")
                        content_to_send_for_llm = f"[Could not extract text from PDF. Original filename: {original_full_filename}]"
                except Exception as e_pdf:
                    print(f"  Error reading PDF {original_full_filename} with pypdf: {e_pdf}")
                    content_to_send_for_llm = f"[Error reading PDF content with pypdf. Original filename: {original_full_filename}]"
            else:
                print("  pypdf library not found or not imported. Please install it: pip install pypdf")
                content_to_send_for_llm = f"[pypdf library not available to read PDF content. Original filename: {original_full_filename}]"
            
            prompt_template = prompts_cfg.get("pdf_analysis", DEFAULT_PROMPTS["pdf_analysis"])
            prompt = prompt_template.format(
                metadata_prompt_section=metadata_prompt_section,
                max_text_chars=max_text_chars_cfg,
                content_to_send_for_llm=content_to_send_for_llm
            )
            print(f"  Analyzing PDF text with {text_model_cfg} (Temp: {temperature_cfg}, Predict: {num_predict_cfg})...")
            response = ollama_client.generate(model=text_model_cfg, prompt=prompt, stream=False, options=ollama_options)
            suggested_base = response['response'].strip()

        elif file_type and (file_type.startswith('text/') or \
             any(file_type.endswith(x) for x in ['document', 'xml', 'json', 'csv', 'markdown', 'python-script', 'javascript'])): 
            if not text_model_cfg:
                print(f"Text model not specified, skipping text file {original_full_filename}")
                return None
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: 
                    content_to_send_for_llm = f.read(max_text_chars_cfg)
                if not content_to_send_for_llm.strip() and os.path.getsize(filepath) > 0: 
                    content_to_send_for_llm = f"[Content is likely binary or unreadable as plain text. Original filename: {original_full_filename}]"
                elif not content_to_send_for_llm.strip(): 
                    content_to_send_for_llm = f"[File content is empty. Original filename: {original_full_filename}]"
            except Exception as e:
                print(f"  Error reading text file {original_full_filename}: {e}")
                content_to_send_for_llm = f"[Error reading text content. Original filename: {original_full_filename}]"
            
            prompt_template = prompts_cfg.get("text_analysis", DEFAULT_PROMPTS["text_analysis"])
            prompt = prompt_template.format(
                metadata_prompt_section=metadata_prompt_section,
                max_text_chars=max_text_chars_cfg,
                content_to_send_for_llm=content_to_send_for_llm
            )
            print(f"  Analyzing text with {text_model_cfg} (Temp: {temperature_cfg}, Predict: {num_predict_cfg})...")
            response = ollama_client.generate(model=text_model_cfg, prompt=prompt, stream=False, options=ollama_options)
            suggested_base = response['response'].strip()
        else:
            print(f"  Skipping unsupported file type: {file_type or 'None'} for {original_full_filename}") 
            return None

        core_suggestion_match = re.search(r'([\w-]+(?:[_-][\w-]+)*)', suggested_base)
        if core_suggestion_match:
            cleaned_suggested_base = core_suggestion_match.group(1)
        else:
            cleaned_suggested_base = re.sub(r'[^a-zA-Z0-9_-]', '', suggested_base.replace(" ", "_"))
        
        cleaned_suggested_base = sanitize_filename_component(cleaned_suggested_base)
        common_llm_phrases = ["sure_heres_a_suggested_filename_base", "heres_a_suggestion", "suggested_filename_base", "a_good_filename_base_would_be", "how_about", "filename_suggestion", "based_on_the_content"]
        for phrase in common_llm_phrases: cleaned_suggested_base = cleaned_suggested_base.replace(phrase, "")
        cleaned_suggested_base = sanitize_filename_component(cleaned_suggested_base.strip('_'))

        if not cleaned_suggested_base:
            print(f"  LLM returned an empty or unusable base name for {original_full_filename} (raw: '{suggested_base}'). Skipping.")
            return None
        
        final_new_name = cleaned_suggested_base + original_ext 
        return final_new_name

    except ollama.ResponseError as e:
        print(f"  Ollama API Error for {original_full_filename}: {e.error}")
        if "model not found" in str(e.error).lower(): 
             model_name_to_check = None
             if file_type and file_type.startswith('image/'): model_name_to_check = vision_model_cfg
             elif file_type and file_type.startswith('video/'): model_name_to_check = vision_model_cfg
             elif file_type and file_type == 'application/pdf': model_name_to_check = text_model_cfg
             else: model_name_to_check = text_model_cfg
             if model_name_to_check:
                 print(f"  Please make sure the model is downloaded: `ollama pull {model_name_to_check}`")
    except Exception as e:
        print(f"  Error analyzing {original_full_filename} with Ollama: {e}")
    return None # Ensure None is returned on any exception path not already returning

def main():
    global CONFIG # Make CONFIG accessible
    load_config() # Load config at the start

    parser = argparse.ArgumentParser(
        description="Rename files in a folder based on their content and metadata using Ollama.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Set defaults from CONFIG, but allow command-line to override
    parser.add_argument("folder", help="Path to the folder containing files to rename.")
    parser.add_argument("--text-model", default=CONFIG.get("models", {}).get("text_model"), help=f"Ollama model for text files (config: {CONFIG.get('models', {}).get('text_model')}).")
    parser.add_argument("--vision-model", default=CONFIG.get("models", {}).get("vision_model"), help=f"Ollama model for image/video frames (config: {CONFIG.get('models', {}).get('vision_model')}).")
    parser.add_argument("--ollama-host", default=CONFIG.get("ollama_host"), help=f"Ollama server address (config: {CONFIG.get('ollama_host')}).")
    parser.add_argument("--dry-run", action="store_true", help="Show proposed renames without actually renaming files.")
    parser.add_argument("--skip-extensions", nargs="*", default=[], help="List of file extensions to skip (e.g., .log .tmp).")
    parser.add_argument("--max-text-chars", type=int, default=CONFIG.get("generation_parameters",{}).get("max_text_chars", 2000), help=f"Max characters from text files for analysis (config: {CONFIG.get('generation_parameters',{}).get('max_text_chars', 2000)}).")
    parser.add_argument("--temperature", type=float, default=CONFIG.get("generation_parameters", {}).get("temperature"), help=f"Temperature for Ollama model generation (config: {CONFIG.get('generation_parameters', {}).get('temperature')}).")
    parser.add_argument("--num-predict", type=int, default=CONFIG.get("generation_parameters", {}).get("num_predict"), help=f"Max tokens to predict (config: {CONFIG.get('generation_parameters', {}).get('num_predict')}).")
    parser.add_argument("--video-frames", type=int, default=CONFIG.get("video_processing", {}).get("frames_to_analyze"), help=f"Number of frames to analyze from video files (config: {CONFIG.get('video_processing', {}).get('frames_to_analyze')}). Set to 0 to skip video analysis.")
    parser.add_argument("--keep-original-name", action="store_true", help="Append suggested name to the original filename instead of replacing it.")
    parser.add_argument("--skip-metadata", action="store_true", help="Do not include file metadata in the LLM prompt.")
    parser.add_argument("-y", "--yes", action="store_true", help="Automatically confirm all renames (use with caution!).")

    args = parser.parse_args()

    # Update CONFIG with command-line arguments if they were provided (and thus differ from default)
    # This ensures that command-line args override config file settings.
    # We check if the arg value is different from the parser's default, which *should* be the config value.
    # A more robust way would be to check if the arg was *actually* passed, but argparse doesn't make this super easy post-parse
    # without custom actions. This approach is generally good enough.

    if args.text_model != CONFIG.get("models", {}).get("text_model"):
        CONFIG.setdefault("models", {})["text_model"] = args.text_model
    if args.vision_model != CONFIG.get("models", {}).get("vision_model"):
        CONFIG.setdefault("models", {})["vision_model"] = args.vision_model
    if args.ollama_host != CONFIG.get("ollama_host"):
        CONFIG["ollama_host"] = args.ollama_host
    if args.temperature != CONFIG.get("generation_parameters", {}).get("temperature"):
        CONFIG.setdefault("generation_parameters", {})["temperature"] = args.temperature
    if args.num_predict != CONFIG.get("generation_parameters", {}).get("num_predict"):
        CONFIG.setdefault("generation_parameters", {})["num_predict"] = args.num_predict
    if args.video_frames != CONFIG.get("video_processing", {}).get("frames_to_analyze"):
        CONFIG.setdefault("video_processing", {})["frames_to_analyze"] = args.video_frames
    if args.max_text_chars != CONFIG.get("generation_parameters",{}).get("max_text_chars", 2000): # max_text_chars wasn't in default config before
        CONFIG.setdefault("generation_parameters", {})["max_text_chars"] = args.max_text_chars


    # Use values from the (potentially updated) CONFIG for the rest of the script
    cfg_text_model = CONFIG.get("models", {}).get("text_model", "").strip() or None
    cfg_vision_model = CONFIG.get("models", {}).get("vision_model", "").strip() or None
    cfg_ollama_host = CONFIG.get("ollama_host")
    cfg_temperature = CONFIG.get("generation_parameters", {}).get("temperature")
    cfg_num_predict = CONFIG.get("generation_parameters", {}).get("num_predict")
    cfg_video_frames = CONFIG.get("video_processing", {}).get("frames_to_analyze")
    cfg_max_text_chars = CONFIG.get("generation_parameters", {}).get("max_text_chars", 2000) # Default if not in config
    prompts = CONFIG.get("prompts", DEFAULT_PROMPTS)


    if not PYPDF_AVAILABLE:
        print("Warning: pypdf library not found. PDF text extraction will be skipped. Install with: pip install pypdf")
    if args.skip_metadata:
        print("INFO: File metadata will NOT be included in LLM prompts.")

    if not cfg_text_model and not cfg_vision_model and cfg_video_frames <= 0 : 
        print("Error: Text model, vision model, and video processing are all disabled. Nothing to do.")
        sys.exit(1)
    # ... (other checks using cfg_ variables)


    global CLIENT
    try:
        CLIENT = ollama.Client(host=cfg_ollama_host)
        CLIENT.list()
        print(f"Successfully connected to Ollama at {cfg_ollama_host}")
    except Exception as e:
        print(f"Error: Could not connect to Ollama at {cfg_ollama_host}. Ensure Ollama is running. Details: {e}")
        sys.exit(1)

    if not os.path.isdir(args.folder):
        print(f"Error: Folder not found: {args.folder}")
        sys.exit(1)

    print(f"Scanning folder: {args.folder}")
    if args.dry_run: print("DRY RUN MODE: No files will actually be renamed.")
    if args.keep_original_name: print("KEEP ORIGINAL NAME MODE: Suggested name will be appended to the original.")

    mimetypes.add_type("video/mts", ".mts", strict=False); mimetypes.add_type("video/mp2t", ".mts", strict=False) 
    mimetypes.add_type("video/mp4", ".mp4", strict=False); mimetypes.add_type("video/mpeg", ".mpeg", strict=False)
    mimetypes.add_type("video/quicktime", ".mov", strict=False); mimetypes.add_type("video/x-msvideo", ".avi", strict=False)
    mimetypes.add_type("video/x-matroska", ".mkv", strict=False); mimetypes.add_type("video/webm", ".webm", strict=False)
    mimetypes.add_type("application/pdf", ".pdf", strict=False); mimetypes.add_type("application/msword", ".doc", strict=False)
    mimetypes.add_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx", strict=False)
    mimetypes.add_type("text/markdown", ".md", strict=False); mimetypes.add_type("text/x-python", ".py", strict=False)
    mimetypes.add_type("application/javascript", ".js", strict=False); mimetypes.add_type("application/json", ".json", strict=False)
    mimetypes.add_type("text/csv", ".csv", strict=False); mimetypes.add_type("text/xml", ".xml", strict=False)

    skipped_extensions = [ext.lower() if ext.startswith('.') else '.' + ext.lower() for ext in args.skip_extensions]
    files_processed = 0; files_renamed = 0
    confirm_all_this_session = args.yes
    known_video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.mpeg', '.mpg', '.mts']

    with tempfile.TemporaryDirectory() as temp_dir_for_frames:
        print(f"Using temporary directory for frames: {temp_dir_for_frames}")
        for filename in os.listdir(args.folder):
            filepath = os.path.join(args.folder, filename)
            if not os.path.isfile(filepath): continue
            original_basename_for_combine, original_extension_for_combine = os.path.splitext(filename) 
            original_ext_lower = original_extension_for_combine.lower()
            if original_ext_lower in skipped_extensions: print(f"Skipping '{filename}' due to extension filter."); continue
            if filename.startswith('.'): print(f"Skipping hidden file: '{filename}'"); continue

            mime_type, _ = mimetypes.guess_type(filepath)
            if original_ext_lower in known_video_extensions:
                if not mime_type or not mime_type.startswith('video/'):
                    new_video_mime = 'video/' + original_ext_lower.lstrip('.'); 
                    if original_ext_lower == '.mts': new_video_mime = 'video/mp2t' 
                    print(f"  Info: Forcing MIME type to '{new_video_mime}' for known video extension '{original_ext_lower}'. Initial guess: {mime_type}"); mime_type = new_video_mime
            elif original_ext_lower == '.pdf': 
                 if not mime_type or mime_type != 'application/pdf':
                    print(f"  Info: Forcing MIME type to 'application/pdf' for .pdf extension. Initial guess: {mime_type}"); mime_type = 'application/pdf'
            
            if mime_type is None: 
                if original_ext_lower in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.log', '.ini', '.cfg', '.yaml', '.yml']: mime_type = 'text/plain'
                elif original_ext_lower in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg']: mime_type = 'image/' + (original_ext_lower[1:] if original_ext_lower.startswith('.') else original_ext_lower)
                elif original_ext_lower in known_video_extensions: mime_type = 'video/' + (original_ext_lower[1:] if original_ext_lower.startswith('.') else original_ext_lower)
                elif original_ext_lower in ['.doc', '.docx']: mime_type = 'application/msword'
                else:
                    print(f"Could not determine MIME type for '{filename}'.")
                    can_process_unknown = (cfg_text_model is not None) 
                    if mime_type is None and can_process_unknown: print(f"  Attempting as generic text file."); mime_type = 'text/plain'
                    elif mime_type is None: print(f"  No suitable model for unknown type '{filename}', skipping."); continue
            
            if mime_type and mime_type.startswith('video/') and cfg_video_frames <= 0:
                print(f"Skipping video analysis for '{filename}' as --video-frames is set to 0."); continue

            print(f"\nProcessing: {filename} (Type: {mime_type or 'unknown'})"); files_processed += 1

            llm_suggested_name_with_ext = get_suggested_name(
                filepath, mime_type, cfg_text_model, cfg_vision_model, CLIENT,
                cfg_max_text_chars, cfg_temperature, cfg_num_predict,
                cfg_video_frames, temp_dir_for_frames, args.skip_metadata, prompts
            )

            if llm_suggested_name_with_ext:
                name_to_process_for_collision = llm_suggested_name_with_ext 
                if args.keep_original_name:
                    llm_suggested_base, _ = os.path.splitext(llm_suggested_name_with_ext)
                    if llm_suggested_base: 
                        name_to_process_for_collision = f"{original_basename_for_combine}_{llm_suggested_base}{original_extension_for_combine}"
                        print(f"  --keep-original-name: Combined name will be '{name_to_process_for_collision}'")
                    else: print(f"  Warning: LLM suggested an empty base name. Using LLM suggestion as is.")
                
                if name_to_process_for_collision.lower() == filename.lower() and name_to_process_for_collision != filename:
                    print(f"  Suggested name '{name_to_process_for_collision}' is a case-only change from '{filename}'.")
                elif name_to_process_for_collision.lower() == filename.lower():
                    print(f"  Suggested name '{name_to_process_for_collision}' is effectively the same as original. Skipping."); continue

                counter = 1; current_name_for_collision_check = name_to_process_for_collision
                final_new_filepath = os.path.join(args.folder, current_name_for_collision_check)
                while os.path.exists(final_new_filepath) and final_new_filepath.lower() != filepath.lower():
                    base_for_collision, ext_for_collision = os.path.splitext(name_to_process_for_collision) 
                    current_name_for_collision_check = f"{base_for_collision}_{counter}{ext_for_collision}"
                    final_new_filepath = os.path.join(args.folder, current_name_for_collision_check); counter += 1
                actual_new_name_after_collision = current_name_for_collision_check 

                if actual_new_name_after_collision.lower() == filename.lower() and actual_new_name_after_collision != filename:
                     print(f"  Suggested name (after collision) '{actual_new_name_after_collision}' is a case-only change from '{filename}'.")
                elif actual_new_name_after_collision.lower() == filename.lower():
                    print(f"  Suggested name (after collision) '{actual_new_name_after_collision}' is effectively the same as original. Skipping."); continue

                print(f"  Original:  {filename}"); print(f"  Suggested: {actual_new_name_after_collision}") 
                if not args.dry_run:
                    perform_rename = confirm_all_this_session
                    if not perform_rename:
                        user_input = input("  Rename? (y/N/s/q) (Yes/No/Yes to all/Quit): ").lower()
                        if user_input == 's': print("  Confirming all subsequent renames for this session."); confirm_all_this_session = True; perform_rename = True
                        elif user_input == 'y': perform_rename = True
                        elif user_input == 'q': print("Quitting."); sys.exit(0)
                    if perform_rename:
                        try: os.rename(filepath, final_new_filepath); print(f"  SUCCESS: Renamed to '{actual_new_name_after_collision}'"); files_renamed +=1
                        except Exception as e: print(f"  ERROR renaming: {e}")
                    else: print("  Skipped by user.")
                else: 
                    if os.path.exists(final_new_filepath) and final_new_filepath.lower() != filepath.lower():
                         print(f"  DRY RUN: Would rename. WARNING: New name '{actual_new_name_after_collision}' (or a variant) might conflict.")
                    else: print(f"  DRY RUN: Would rename '{filename}' to '{actual_new_name_after_collision}'")
    print(f"\nFile renaming process complete. Processed: {files_processed} files. Renamed: {files_renamed} files.")

if __name__ == "__main__":
    main()
