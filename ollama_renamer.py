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
import json # For parsing ffprobe output

# Attempt to import pypdf for PDF text extraction
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    PdfReader = None # To avoid NameError if not available

# Global Ollama client
CLIENT = None

def sanitize_filename_component(component):
    """
    Sanitizes a single filename component (base name or extension).
    Removes leading/trailing whitespace, quotes.
    Replaces spaces with underscores.
    Removes characters not suitable for filenames.
    Converts to lowercase.
    """
    name = component.strip().strip("'").strip('"')
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^\w._-]', '', name) # Allow alphanumeric, underscore, hyphen, dot
    return name.lower()

def get_file_metadata(filepath):
    """
    Gathers basic metadata for a file.
    """
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
        # For videos, try to add duration using ffprobe
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
    """
    Gets video duration using ffprobe and returns it as a string or None.
    """
    try:
        cmd = [ 
            'ffprobe', 
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration_seconds = float(result.stdout.strip())
        td = datetime.timedelta(seconds=duration_seconds)
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_formatted = ""
        if td.days > 0: 
             duration_formatted += f"{td.days}d "
        if hours > 0:
            duration_formatted += f"{hours}h "
        if minutes > 0:
            duration_formatted += f"{minutes}m "
        duration_formatted += f"{seconds}s"
        return duration_formatted.strip()
    except subprocess.CalledProcessError as e:
        print(f"  ffprobe error getting duration for {video_filepath}: {e.stderr}")
        return None
    except FileNotFoundError:
        print("  Error: ffprobe command not found. Is FFmpeg installed and in PATH?")
        return None
    except Exception as e:
        print(f"  Unexpected error getting video duration for {video_filepath}: {e}")
        return None


def extract_frames_as_base64(video_filepath, num_frames_to_extract, temp_dir):
    """
    Extracts a specified number of frames evenly spaced throughout the video,
    saves them as temporary PNGs, converts to base64, and returns a list of base64 strings.
    Deletes temporary files afterwards.
    """
    base64_frames = []
    temp_frame_files = []

    try:
        cmd_duration = [ 
            'ffprobe', 
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_filepath
        ]
        result_duration = subprocess.run(cmd_duration, capture_output=True, text=True, check=True)
        duration = float(result_duration.stdout.strip())

        if duration <= 0:
            print(f"  Video {video_filepath} has zero or negative duration. Skipping frame extraction.")
            return []

        if num_frames_to_extract <= 0:
            num_frames_to_extract = 1 

        interval = duration / (num_frames_to_extract + 1) 
        timestamps = [interval * (i + 1) for i in range(num_frames_to_extract)]
        
        if not timestamps and duration > 0 : 
            timestamps = [duration / 2] 
        elif not timestamps and duration <=0 : 
             print(f"  Video {video_filepath} has zero or negative duration after check. Skipping frame extraction.")
             return []

        print(f"  Extracting {len(timestamps)} frames from {os.path.basename(video_filepath)} (duration: {duration:.2f}s)...")

        for i, ts in enumerate(timestamps):
            temp_frame_path = os.path.join(temp_dir, f"frame_{i+1}_{os.path.basename(video_filepath)}.png")
            temp_frame_files.append(temp_frame_path)

            cmd_extract = [ 
                'ffmpeg', 
                '-ss', str(ts),
                '-i', video_filepath,
                '-frames:v', '1',
                '-q:v', '2', 
                '-y', 
                temp_frame_path
            ]
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
        return [] 
    except FileNotFoundError:
        print("  Error: ffmpeg or ffprobe command not found. Is FFmpeg installed and in PATH?")
        return []
    except Exception as e:
        print(f"  Unexpected error during frame extraction for {video_filepath}: {e}")
        return []
    finally:
        for temp_file in temp_frame_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e_del:
                    print(f"  Warning: Could not delete temporary frame file {temp_file}: {e_del}")
    
    return base64_frames


def get_suggested_name(filepath, file_type, text_model, vision_model, ollama_client,
                       max_text_chars=2000, temperature=0.4, num_predict=70,
                       video_frames_to_analyze=5, temp_dir_for_frames=None): 
    original_full_filename = os.path.basename(filepath)
    original_basename, original_ext = os.path.splitext(original_full_filename)
    original_ext = original_ext.lower()

    file_metadata_info = get_file_metadata(filepath)
    ollama_options = {"temperature": temperature, "num_predict": num_predict}
    suggested_base = ""
    content_to_send_for_llm = "" # Initialize content to send to LLM

    try:
        if file_type and file_type.startswith('image/'): 
            if not vision_model:
                print(f"Vision model not specified, skipping image {original_full_filename}")
                return None
            with open(filepath, 'rb') as image_file:
                img_b64 = base64.b64encode(image_file.read()).decode('utf-8')
            prompt = (
                f"Analyze this image and its metadata. Suggest a detailed and descriptive filename base "
                f"(lowercase, underscores for spaces, no file extension). "
                f"If the image features a recurring subject or scene element (e.g., a specific person, pet, location like 'my_backyard_bird_feeder'), "
                f"try to use a consistent and specific term for that element in the filename. " # Added instruction for consistency
                f"For example, for a series of photos of the same bird feeder, always try to include 'backyard_bird_feeder' or a similar specific, consistent term. "
                f"Then add specific details for this particular image. "
                f"Example for a general scene: 'ginger_cat_sleeping_blue_armchair_sunlit_room'. Avoid generic names.\n\n"
                f"File Metadata:\n{file_metadata_info}\n\n"
                f"Only output the suggested filename base."
            )
            print(f"  Analyzing image with {vision_model} (Temp: {temperature}, Predict: {num_predict})...")
            response = ollama_client.generate(model=vision_model, prompt=prompt, images=[img_b64], stream=False, options=ollama_options)
            suggested_base = response['response'].strip()

        elif file_type and file_type.startswith('video/'): 
            if not vision_model or not text_model:
                print(f"  Vision or Text model not specified, skipping video {original_full_filename}")
                return None
            if not temp_dir_for_frames:
                print(f"  Temporary directory for frames not provided. Skipping video {original_full_filename}")
                return None

            print(f"  Processing video: {original_full_filename}")
            b64_frames = extract_frames_as_base64(filepath, video_frames_to_analyze, temp_dir_for_frames)

            if not b64_frames:
                print(f"  No frames extracted or analyzed for video {original_full_filename}. Skipping.")
                return None

            frame_descriptions_content_only = [] 
            print(f"  Analyzing {len(b64_frames)} frames with {vision_model}...")
            for i, frame_b64 in enumerate(b64_frames):
                # The prompt for individual frames can also benefit from consistency hints if applicable,
                # but the main consistency will come from the summary stage.
                prompt_frame = (
                    f"This is frame {i+1} of {len(b64_frames)} from a video. "
                    f"Describe this video frame in detail. Focus on key subjects, actions, and setting. "
                    f"If a clear, recurring subject is visible (e.g., a specific type of bird, a particular event), mention it consistently.\n\n"
                    f"File Metadata (for context of whole video):\n{file_metadata_info}\n\n"
                    f"Only output the description of this single frame."
                )
                try:
                    response_frame = ollama_client.generate(
                        model=vision_model,
                        prompt=prompt_frame,
                        images=[frame_b64],
                        stream=False,
                        options=ollama_options 
                    )
                    description = response_frame['response'].strip()
                    if description:
                        frame_descriptions_content_only.append(description) 
                    print(f"    Frame {i+1} description: {description[:100]}...") 
                except Exception as e_frame:
                    print(f"    Error analyzing frame {i+1} for {original_full_filename}: {e_frame}")
            
            if not frame_descriptions_content_only:
                print(f"  No valid descriptions obtained from frames for {original_full_filename}. Skipping.")
                return None

            combined_descriptions_for_summary = "\n".join(frame_descriptions_content_only)
            
            prompt_summary = (
                f"The following are sequential descriptions of scenes from a video. "
                f"Based *only* on the content of these descriptions, suggest a detailed and descriptive filename base "
                f"(lowercase, underscores for spaces, no file extension). "
                f"Focus on the main theme, subject, or overall progression depicted in these scenes. "
                f"If a recurring subject or theme is evident across multiple descriptions (e.g., 'bird_feeder', 'specific_event_name'), " # Added hint for summary
                f"ensure this recurring element is part of the filename base for consistency. "
                f"Do NOT include literal text like 'frame 1' or 'scene description' in the filename itself. "
                f"Avoid generic names.\n\n"
                f"File Metadata (for context of whole video):\n{file_metadata_info}\n\n"
                f"Sequential Scene Descriptions (ignore any 'Frame X:' labels if they accidentally appear here, focus on the actual described content):\n{combined_descriptions_for_summary}\n\n"
                f"Only output the suggested filename base."
            )
            print(f"  Summarizing frame descriptions with {text_model}...")
            response_summary = ollama_client.generate(
                model=text_model,
                prompt=prompt_summary,
                stream=False,
                options=ollama_options
            )
            suggested_base = response_summary['response'].strip()
        
        # PDF Handling Block
        elif file_type and file_type == 'application/pdf':
            if not text_model:
                print(f"Text model not specified, skipping PDF file {original_full_filename}")
                return None
            print(f"  Processing PDF: {original_full_filename}")
            if PYPDF_AVAILABLE and PdfReader:
                try:
                    reader = PdfReader(filepath)
                    extracted_text = ""
                    # Limit number of pages to read to prevent very long processing for huge PDFs
                    num_pages_to_read = min(len(reader.pages), 50) # Read up to 50 pages
                    for page_num in range(num_pages_to_read):
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n" 
                        if len(extracted_text) >= max_text_chars:
                            break
                    if extracted_text.strip():
                        content_to_send_for_llm = extracted_text[:max_text_chars]
                        print(f"  Successfully extracted ~{len(content_to_send_for_llm)} chars of text from PDF (up to {num_pages_to_read} pages).")
                    else:
                        print(f"  Could not extract text from PDF (it might be image-based, empty, or encrypted).")
                        content_to_send_for_llm = f"[Could not extract text from PDF. Metadata hints might be useful. Original filename: {original_full_filename}]"
                except Exception as e_pdf:
                    print(f"  Error reading PDF {original_full_filename} with pypdf: {e_pdf}")
                    content_to_send_for_llm = f"[Error reading PDF content with pypdf. Original filename: {original_full_filename}]"
            else:
                print("  pypdf library not found or not imported. Please install it: pip install pypdf")
                content_to_send_for_llm = f"[pypdf library not available to read PDF content. Original filename: {original_full_filename}]"
            
            prompt = (
                f"Analyze the following text content (extracted from a PDF) and file metadata. "
                f"Suggest a detailed and descriptive filename base (lowercase, underscores for spaces, no file extension). "
                f"Example: 'project_phoenix_q3_marketing_strategy_meeting_notes'. Avoid generic names.\n\n"
                f"File Metadata:\n{file_metadata_info}\n\n"
                f"PDF Text content (first {max_text_chars} chars, if available):\n{content_to_send_for_llm}\n\n"
                f"Only output the suggested filename base."
            )
            print(f"  Analyzing PDF text with {text_model} (Temp: {temperature}, Predict: {num_predict})...")
            response = ollama_client.generate(model=text_model, prompt=prompt, stream=False, options=ollama_options)
            suggested_base = response['response'].strip()

        # General Text File Handling (excluding PDF, which is now handled above)
        elif file_type and (file_type.startswith('text/') or \
             any(file_type.endswith(x) for x in ['document', 'xml', 'json', 'csv', 'markdown', 'python-script', 'javascript'])): 
            if not text_model:
                print(f"Text model not specified, skipping text file {original_full_filename}")
                return None
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: 
                    content_to_send_for_llm = f.read(max_text_chars)
                if not content_to_send_for_llm.strip() and os.path.getsize(filepath) > 0: 
                    content_to_send_for_llm = f"[Content is likely binary or unreadable as plain text. Metadata hints might be useful. Original filename: {original_full_filename}]"
                elif not content_to_send_for_llm.strip(): 
                    content_to_send_for_llm = f"[File content is empty. Rely on metadata. Original filename: {original_full_filename}]"
            except Exception as e:
                print(f"  Error reading text file {original_full_filename}: {e}")
                content_to_send_for_llm = f"[Error reading text content. Original filename: {original_full_filename}]"
            
            prompt = (
                f"Analyze the following text content and file metadata. "
                f"Suggest a detailed and descriptive filename base (lowercase, underscores for spaces, no file extension). "
                f"Example: 'project_phoenix_q3_marketing_strategy_meeting_notes'. Avoid generic names.\n\n"
                f"File Metadata:\n{file_metadata_info}\n\n"
                f"Text content (first {max_text_chars} chars, if available):\n{content_to_send_for_llm}\n\n"
                f"Only output the suggested filename base."
            )
            print(f"  Analyzing text with {text_model} (Temp: {temperature}, Predict: {num_predict})...")
            response = ollama_client.generate(model=text_model, prompt=prompt, stream=False, options=ollama_options)
            suggested_base = response['response'].strip()
        else:
            print(f"  Skipping unsupported file type: {file_type or 'None'} for {original_full_filename}") 
            return None

        # Common cleaning logic for suggested_base from any source
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
             if file_type and file_type.startswith('image/'): model_name_to_check = vision_model
             elif file_type and file_type.startswith('video/'): model_name_to_check = vision_model 
             elif file_type and file_type == 'application/pdf': model_name_to_check = text_model
             else: model_name_to_check = text_model # General text
             if model_name_to_check:
                 print(f"  Please make sure the model is downloaded: `ollama pull {model_name_to_check}`")
        return None
    except Exception as e:
        print(f"  Error analyzing {original_full_filename} with Ollama: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Rename files in a folder based on their content and metadata using Ollama.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("folder", help="Path to the folder containing files to rename.")
    parser.add_argument("--text-model", default="gemma:2b", help="Ollama model for text files (e.g., gemma:2b).")
    parser.add_argument("--vision-model", default="llava:latest", help="Ollama model for image/video frames (e.g., llava).")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama server address.")
    parser.add_argument("--dry-run", action="store_true", help="Show proposed renames without actually renaming files.")
    parser.add_argument("--skip-extensions", nargs="*", default=[], help="List of file extensions to skip (e.g., .log .tmp).")
    parser.add_argument("--max-text-chars", type=int, default=2000, help="Max characters from text files for analysis.")
    parser.add_argument("--temperature", type=float, default=0.4, help="Temperature for Ollama model generation.")
    parser.add_argument("--num-predict", type=int, default=70, help="Max tokens to predict (Ollama's num_predict).")
    parser.add_argument("--video-frames", type=int, default=5, help="Number of frames to analyze from video files (default: 5). Set to 0 to skip video analysis.")
    parser.add_argument("--keep-original-name", action="store_true", help="Append suggested name to the original filename instead of replacing it.")
    parser.add_argument("-y", "--yes", action="store_true", help="Automatically confirm all renames (use with caution!).")

    args = parser.parse_args()

    args.text_model = args.text_model.strip() if args.text_model else None
    args.vision_model = args.vision_model.strip() if args.vision_model else None

    if not PYPDF_AVAILABLE:
        print("Warning: pypdf library not found. PDF text extraction will be skipped. Install with: pip install pypdf")


    if not args.text_model and not args.vision_model and args.video_frames <= 0 : 
        print("Error: Text model, vision model, and video processing are all disabled. Nothing to do.")
        sys.exit(1)
    if args.video_frames > 0 and not args.vision_model:
        print("Warning: --video-frames specified, but --vision-model is disabled. Video frames will not be analyzed by vision model.")
    if args.video_frames > 0 and not args.text_model: 
        print("Warning: --video-frames specified, but --text-model is disabled. Video frame descriptions will not be summarized by text model.")
    if not args.text_model and PYPDF_AVAILABLE:
        print("Warning: pypdf is available for PDF processing, but --text-model is disabled. PDF text will not be analyzed by LLM.")


    global CLIENT
    try:
        CLIENT = ollama.Client(host=args.ollama_host)
        CLIENT.list()
        print(f"Successfully connected to Ollama at {args.ollama_host}")
    except Exception as e:
        print(f"Error: Could not connect to Ollama at {args.ollama_host}. Ensure Ollama is running. Details: {e}")
        sys.exit(1)

    if not os.path.isdir(args.folder):
        print(f"Error: Folder not found: {args.folder}")
        sys.exit(1)

    print(f"Scanning folder: {args.folder}")
    if args.dry_run:
        print("DRY RUN MODE: No files will actually be renamed.")
    if args.keep_original_name:
        print("KEEP ORIGINAL NAME MODE: Suggested name will be appended to the original.")


    mimetypes.add_type("video/mts", ".mts", strict=False) 
    mimetypes.add_type("video/mp2t", ".mts", strict=False) 
    mimetypes.add_type("video/mp4", ".mp4", strict=False)
    mimetypes.add_type("video/mpeg", ".mpeg", strict=False)
    mimetypes.add_type("video/quicktime", ".mov", strict=False)
    mimetypes.add_type("video/x-msvideo", ".avi", strict=False)
    mimetypes.add_type("video/x-matroska", ".mkv", strict=False)
    mimetypes.add_type("video/webm", ".webm", strict=False)
    mimetypes.add_type("application/pdf", ".pdf", strict=False) 
    mimetypes.add_type("application/msword", ".doc", strict=False)
    mimetypes.add_type("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx", strict=False)
    mimetypes.add_type("text/markdown", ".md", strict=False)
    mimetypes.add_type("text/x-python", ".py", strict=False)
    mimetypes.add_type("application/javascript", ".js", strict=False)
    mimetypes.add_type("application/json", ".json", strict=False)
    mimetypes.add_type("text/csv", ".csv", strict=False)
    mimetypes.add_type("text/xml", ".xml", strict=False)


    skipped_extensions = [ext.lower() if ext.startswith('.') else '.' + ext.lower() for ext in args.skip_extensions]
    files_processed = 0
    files_renamed = 0
    confirm_all_this_session = args.yes
    
    known_video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.mpeg', '.mpg', '.mts']


    with tempfile.TemporaryDirectory() as temp_dir_for_frames:
        print(f"Using temporary directory for frames: {temp_dir_for_frames}")
        for filename in os.listdir(args.folder):
            filepath = os.path.join(args.folder, filename)
            if not os.path.isfile(filepath): continue

            original_basename_for_combine, original_extension_for_combine = os.path.splitext(filename) 
            original_ext_lower = original_extension_for_combine.lower()

            if original_ext_lower in skipped_extensions:
                print(f"Skipping '{filename}' due to extension filter.")
                continue
            if filename.startswith('.'):
                print(f"Skipping hidden file: '{filename}'")
                continue

            mime_type, _ = mimetypes.guess_type(filepath)

            if original_ext_lower in known_video_extensions:
                if not mime_type or not mime_type.startswith('video/'):
                    new_video_mime = 'video/' + original_ext_lower.lstrip('.')
                    if original_ext_lower == '.mts':
                        new_video_mime = 'video/mp2t' 
                    print(f"  Info: Forcing MIME type to '{new_video_mime}' for known video extension '{original_ext_lower}'. Initial guess: {mime_type}")
                    mime_type = new_video_mime
            elif original_ext_lower == '.pdf': 
                 if not mime_type or mime_type != 'application/pdf':
                    print(f"  Info: Forcing MIME type to 'application/pdf' for .pdf extension. Initial guess: {mime_type}")
                    mime_type = 'application/pdf'

            
            if mime_type is None: 
                if original_ext_lower in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv', '.log', '.ini', '.cfg', '.yaml', '.yml']: mime_type = 'text/plain'
                elif original_ext_lower in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg']: mime_type = 'image/' + (original_ext_lower[1:] if original_ext_lower.startswith('.') else original_ext_lower)
                elif original_ext_lower in known_video_extensions: mime_type = 'video/' + (original_ext_lower[1:] if original_ext_lower.startswith('.') else original_ext_lower)
                elif original_ext_lower in ['.doc', '.docx']: mime_type = 'application/msword'
                else:
                    print(f"Could not determine MIME type for '{filename}'.")
                    can_process_unknown = (args.text_model is not None) 
                    if mime_type is None and can_process_unknown:
                        print(f"  Attempting as generic text file.")
                        mime_type = 'text/plain'
                    elif mime_type is None:
                        print(f"  No suitable model for unknown type '{filename}', skipping.")
                        continue
            
            if mime_type and mime_type.startswith('video/') and args.video_frames <= 0:
                print(f"Skipping video analysis for '{filename}' as --video-frames is set to 0.")
                continue


            print(f"\nProcessing: {filename} (Type: {mime_type or 'unknown'})")
            files_processed += 1

            llm_suggested_name_with_ext = get_suggested_name(
                filepath, mime_type, args.text_model, args.vision_model, CLIENT,
                args.max_text_chars, args.temperature, args.num_predict,
                args.video_frames, temp_dir_for_frames 
            )

            if llm_suggested_name_with_ext:
                name_to_process_for_collision = llm_suggested_name_with_ext 

                if args.keep_original_name:
                    llm_suggested_base, _ = os.path.splitext(llm_suggested_name_with_ext)
                    if llm_suggested_base: 
                        name_to_process_for_collision = f"{original_basename_for_combine}_{llm_suggested_base}{original_extension_for_combine}"
                        print(f"  --keep-original-name: Combined name will be '{name_to_process_for_collision}'")
                    else:
                        print(f"  Warning: LLM suggested an empty base name from '{llm_suggested_name_with_ext}'. Using LLM suggestion as is for keep_original_name.")
                
                if name_to_process_for_collision.lower() == filename.lower() and name_to_process_for_collision != filename:
                    print(f"  Suggested name '{name_to_process_for_collision}' is a case-only change from '{filename}'.")
                elif name_to_process_for_collision.lower() == filename.lower():
                    print(f"  Suggested name '{name_to_process_for_collision}' is effectively the same as original. Skipping.")
                    continue

                counter = 1
                current_name_for_collision_check = name_to_process_for_collision
                final_new_filepath = os.path.join(args.folder, current_name_for_collision_check)

                while os.path.exists(final_new_filepath) and final_new_filepath.lower() != filepath.lower():
                    base_for_collision, ext_for_collision = os.path.splitext(name_to_process_for_collision) 
                    current_name_for_collision_check = f"{base_for_collision}_{counter}{ext_for_collision}"
                    final_new_filepath = os.path.join(args.folder, current_name_for_collision_check)
                    counter += 1
                
                actual_new_name_after_collision = current_name_for_collision_check 

                if actual_new_name_after_collision.lower() == filename.lower() and actual_new_name_after_collision != filename:
                     print(f"  Suggested name (after collision) '{actual_new_name_after_collision}' is a case-only change from '{filename}'.")
                elif actual_new_name_after_collision.lower() == filename.lower():
                    print(f"  Suggested name (after collision) '{actual_new_name_after_collision}' is effectively the same as original. Skipping.")
                    continue

                print(f"  Original:  {filename}")
                print(f"  Suggested: {actual_new_name_after_collision}") 

                if not args.dry_run:
                    perform_rename = confirm_all_this_session
                    if not perform_rename:
                        user_input = input("  Rename? (y/N/s/q) (Yes/No/Yes to all/Quit): ").lower()
                        if user_input == 's':
                            print("  Confirming all subsequent renames for this session.")
                            confirm_all_this_session = True; perform_rename = True
                        elif user_input == 'y': perform_rename = True
                        elif user_input == 'q': print("Quitting."); sys.exit(0)
                    if perform_rename:
                        try:
                            os.rename(filepath, final_new_filepath) 
                            print(f"  SUCCESS: Renamed to '{actual_new_name_after_collision}'")
                            files_renamed +=1
                        except Exception as e: print(f"  ERROR renaming: {e}")
                    else: print("  Skipped by user.")
                else: 
                    if os.path.exists(final_new_filepath) and final_new_filepath.lower() != filepath.lower():
                         print(f"  DRY RUN: Would rename. WARNING: New name '{actual_new_name_after_collision}' (or a variant due to collision) might conflict with an existing different file.")
                    else: print(f"  DRY RUN: Would rename '{filename}' to '{actual_new_name_after_collision}'")


    print(f"\nFile renaming process complete. Processed: {files_processed} files. Renamed: {files_renamed} files.")

if __name__ == "__main__":
    main()
