# renamer_core.py
import ollama
import os
import base64
import mimetypes
import re
import sys
import datetime
import subprocess
import tempfile
import json
import traceback

# Attempt to import pypdf for PDF text extraction
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    PdfReader = None

def log_message(message, log_callback):
    """Utility to print messages and send them to a GUI callback if available."""
    if log_callback:
        log_callback(message + '\n')
    else:
        print(message)

def extract_json_from_text(text):
    """Extracts a JSON object from a string."""
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"  JSON parse failed: {e}")
            return None
    return None

def sanitize_filename_component(component):
    """Cleans a string to be a valid filename component."""
    name = component.strip().strip("'").strip('"')
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'[^\w._-]', '', name)
    return name.lower()

def get_file_metadata(filepath, skip_metadata):
    """Gathers metadata for a given file path."""
    if skip_metadata:
        return ""
    try:
        stat_info = os.stat(filepath)
        creation_time = datetime.datetime.fromtimestamp(stat_info.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        mod_time = datetime.datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        size_kb = round(stat_info.st_size / 1024, 2)
        metadata_str = f"Original: '{os.path.basename(filepath)}', Created: {creation_time}, Modified: {mod_time}, Size: {size_kb}KB"
        
        mime_type, _ = mimetypes.guess_type(filepath)
        ext = os.path.splitext(filepath)[1].lower()
        if (mime_type and mime_type.startswith('video/')) or ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.mpeg', '.mpg', '.mts']:
            duration = get_video_duration_str(filepath)
            if duration:
                metadata_str += f", Duration: {duration}"
        return metadata_str
    except Exception as e:
        return f"Original filename: '{os.path.basename(filepath)}', Error getting metadata: {e}"

def get_video_duration_str(video_filepath):
    """Returns the duration of a video file as a formatted string."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_filepath]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, creationflags=flags)
        seconds = float(result.stdout.strip())
        td = datetime.timedelta(seconds=seconds)
        h, r = divmod(td.seconds, 3600)
        m, s = divmod(r, 60)
        return f"{f'{td.days}d ' if td.days>0 else ''}{f'{h}h ' if h>0 else ''}{f'{m}m ' if m>0 else ''}{s}s"
    except Exception:
        return None

def extract_frames_as_base64(video_filepath, num_frames, temp_dir, log_callback):
    """Extracts frames from a video and returns them as a list of base64 strings."""
    base64_frames = []
    temp_files = []
    try:
        cmd_duration = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_filepath]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        result_duration = subprocess.run(cmd_duration, capture_output=True, text=True, check=True, creationflags=flags)
        duration = float(result_duration.stdout.strip())

        if duration <= 0: return []
        if num_frames <= 0: num_frames = 1
        
        interval = duration / (num_frames + 1)
        timestamps = [interval * (i + 1) for i in range(num_frames)]
        if not timestamps and duration > 0: timestamps = [duration / 2]
        if not timestamps: return []
        
        log_message(f"  Extracting {len(timestamps)} frames from {os.path.basename(video_filepath)}...", log_callback)
        for i, ts in enumerate(timestamps):
            path = os.path.join(temp_dir, f"frame_{i+1}.png")
            temp_files.append(path)
            cmd_extract = ['ffmpeg', '-ss', str(ts), '-i', video_filepath, '-frames:v', '1', '-q:v', '2', '-y', path]
            p = subprocess.run(cmd_extract, capture_output=True, text=True, creationflags=flags)
            if p.returncode != 0:
                log_message(f"  FFmpeg error at {ts:.2f}s: {p.stderr}", log_callback)
                continue
            if os.path.exists(path) and os.path.getsize(path) > 0:
                with open(path, 'rb') as f:
                    base64_frames.append(base64.b64encode(f.read()).decode('utf-8'))
    except Exception as e:
        log_message(f"  Error during frame extraction: {e}", log_callback)
    finally:
        for f in temp_files:
            if os.path.exists(f):
                try: os.remove(f)
                except Exception: pass
    return base64_frames

def get_suggested_name(client, filepath, file_type, cfg, skip_metadata_flag, temp_dir, stop_flag_check, log_callback):
    """Gets a suggested filename from Ollama based on file content."""
    if stop_flag_check(): return None
    
    original_full_filename = os.path.basename(filepath)
    original_ext = os.path.splitext(original_full_filename)[1].lower()
    file_metadata_info = get_file_metadata(filepath, skip_metadata_flag)
    ollama_options = {
        "temperature": cfg["generation_parameters"]["temperature"],
        "num_predict": cfg["generation_parameters"]["num_predict"]
    }
    metadata_prompt_section = f"File Metadata:\n{file_metadata_info}\n\n" if file_metadata_info else ""
    suggested_base = ""

    try:
        if file_type and file_type.startswith('image/'):
            vision_model = cfg["models"]["vision_model"]
            if not vision_model: log_message("Vision model not set...", log_callback); return None
            with open(filepath, 'rb') as f: img_b64 = base64.b64encode(f.read()).decode('utf-8')
            prompt = cfg["prompts"]["image_analysis"].replace('{metadata_prompt_section}', metadata_prompt_section)
            log_message(f"  Analyzing image with {vision_model}...", log_callback)
            response = client.generate(model=vision_model, prompt=prompt, images=[img_b64], stream=False, options=ollama_options)
            suggested_base = response['response'].strip()

        elif file_type and file_type.startswith('video/'):
            vision_model = cfg["models"]["vision_model"]
            text_model = cfg["models"]["text_model"]
            if not vision_model or not text_model: log_message("Vision or Text model not set...", log_callback); return None
            frames_to_analyze = cfg["video_processing"]["frames_to_analyze"]
            b64_frames = extract_frames_as_base64(filepath, frames_to_analyze, temp_dir, log_callback)
            if not b64_frames: log_message("  No frames extracted.", log_callback); return None
            
            frame_descriptions = []
            frame_prompt_template = cfg["prompts"]["video_frame_analysis"]
            for i, frame_b64 in enumerate(b64_frames):
                if stop_flag_check(): return None
                prompt = frame_prompt_template.replace('{metadata_prompt_section}', metadata_prompt_section)
                response = client.generate(model=vision_model, prompt=prompt, images=[frame_b64], stream=False, options=ollama_options)
                desc = response['response'].strip()
                frame_descriptions.append(desc)
                log_message(f"    Frame {i+1}: {desc[:100]}...", log_callback)
            
            if not frame_descriptions: log_message("  No descriptions from frames.", log_callback); return None
            summary_prompt_template = cfg["prompts"]["video_summary"]
            summary_prompt = summary_prompt_template.replace('{metadata_prompt_section}', metadata_prompt_section).replace('{combined_descriptions}', "\n".join(frame_descriptions))
            log_message(f"  Summarizing with {text_model}...", log_callback)
            response = client.generate(model=text_model, prompt=summary_prompt, stream=False, options=ollama_options)
            suggested_base = response['response'].strip()

        elif (file_type and file_type == 'application/pdf') or (file_type and file_type.startswith('text/')):
            text_model = cfg["models"]["text_model"]
            max_text_chars = cfg["generation_parameters"]["max_text_chars"]
            if not text_model: log_message("Text model not set...", log_callback); return None
            content = ""
            is_pdf = file_type == 'application/pdf'
            
            if is_pdf:
                if PYPDF_AVAILABLE:
                    try:
                        reader = PdfReader(filepath)
                        for page in reader.pages[:50]: content += page.extract_text() + "\n"
                        content = content[:max_text_chars]
                        if not content.strip(): content = "[PDF content is empty or image-based]"
                    except Exception as e: content = f"[Error reading PDF: {e}]"
                else: content = "[pypdf library not installed]"
            else:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: content = f.read(max_text_chars)
            
            prompt_key = "pdf_analysis" if is_pdf else "text_analysis"
            prompt_template = cfg["prompts"][prompt_key]
            prompt = prompt_template.replace('{metadata_prompt_section}', metadata_prompt_section).replace('{max_text_chars}', str(max_text_chars)).replace('{content_to_send_for_llm}', content)
            response = client.generate(model=text_model, prompt=prompt, stream=False, options=ollama_options)
            suggested_base = response['response'].strip()
            
        else:
            log_message(f"  Skipping unsupported file type: {file_type}", log_callback)
            return None

        if not suggested_base: log_message("  LLM returned empty suggestion.", log_callback); return None
        
        base_without_ext, ext = os.path.splitext(suggested_base)
        if ext and len(ext) > 1 and len(ext) < 6 : # Check for plausible extension
             suggested_base = base_without_ext

        common_phrases = ["sure", "heres_a_suggestion", "suggested_filename_base", "a_good_filename_base_would_be", "how_about",
                          "filename_suggestion", "based_on_the_content", "filename_base", "the_filename", "a_filename",
                          "a_descriptive_filename", "descriptive_filename", "certainly", "the", "a", "an", "is", "of"]
        cleaned_base = sanitize_filename_component(suggested_base)
        words = cleaned_base.split('_')
        filtered_words = [word for word in words if word not in common_phrases and word]
        cleaned_base = '_'.join(filtered_words)
        
        if not cleaned_base: log_message(f"  Warning: Suggestion '{suggested_base}' was reduced to nothing after cleaning.", log_callback); return None
        
        return cleaned_base + original_ext

    except Exception as e:
        log_message(f"  ERROR processing {original_full_filename}: {e}", log_callback)
        traceback.print_exc()
        return None

def process_directory(target_folder, gui_opts, config, stop_flag_check, log_callback):
    """The main logic for analyzing, grouping, and renaming files in a directory."""
    try:
        client = ollama.Client(host=config["ollama_host"])
        client.list()
        log_message(f"Connected to Ollama at {config['ollama_host']}", log_callback)
    except Exception as e:
        log_message(f"Error connecting to Ollama: {e}", log_callback)
        return

    # PASS 1: ANALYSIS
    log_message("\n--- Starting Pass 1: Analyzing all files ---", log_callback)
    analysis_results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for filename in sorted(os.listdir(target_folder)):
            if stop_flag_check(): log_message("Stopping analysis.", log_callback); break
            filepath = os.path.join(target_folder, filename)
            if not os.path.isfile(filepath): continue
            
            _, original_ext = os.path.splitext(filename)
            mime_type, _ = mimetypes.guess_type(filepath)
            
            if original_ext.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.mpeg', '.mpg', '.mts']: mime_type = 'video/'
            elif original_ext.lower() == '.pdf': mime_type = 'application/pdf'
                
            if not mime_type: log_message(f"Skipping unknown file type: {filename}", log_callback); continue
            if mime_type.startswith('video/') and config["video_processing"]["frames_to_analyze"] <= 0: log_message(f"Skipping video: {filename}", log_callback); continue

            log_message(f"\nProcessing: {filename} (Type: {mime_type})", log_callback)
            suggested_name = get_suggested_name(client, filepath, mime_type, config, gui_opts["skip_metadata"], temp_dir, stop_flag_check, log_callback)
            
            if suggested_name:
                analysis_results.append({
                    "original_path": filepath, "original_name": filename,
                    "suggested_name": suggested_name, "processed": False
                })
    if stop_flag_check(): return

    # PASS 2: GROUPING (OPTIONAL)
    final_rename_plan = {}
    if gui_opts["group_similar"] and analysis_results:
        log_message("\n--- Starting Pass 2: Grouping similar filenames ---", log_callback)
        suggested_names = [res["suggested_name"] for res in analysis_results]
        prompt_template = config["prompts"]["filename_grouping"]
        prompt = prompt_template.replace('{filename_list_json}', json.dumps(suggested_names))
        try:
            system_prompt = "You are an expert file organizer. You only respond with valid, raw JSON. Do not include any other text or formatting."
            response = client.generate(model=config["models"]["text_model"], system=system_prompt, prompt=prompt, stream=False)
            grouped_data = extract_json_from_text(response['response'])
            
            if not grouped_data or not isinstance(grouped_data, dict):
                raise ValueError("Could not parse a valid JSON object from the LLM response for grouping.")
            
            log_message("  Successfully parsed JSON response from LLM.", log_callback)
            for uniform_base, original_list in grouped_data.items():
                safe_uniform_base = sanitize_filename_component(uniform_base)
                if len(original_list) > 1:
                    for i, original_sugg_name in enumerate(sorted(original_list)):
                        for res in analysis_results:
                            if res["suggested_name"] == original_sugg_name and not res["processed"]:
                                _, ext = os.path.splitext(original_sugg_name)
                                new_grouped_name = f"{safe_uniform_base}_{i+1:03d}{ext}"
                                final_rename_plan[res["original_path"]] = new_grouped_name
                                log_message(f"  Grouping '{original_sugg_name}' -> '{new_grouped_name}'", log_callback)
                                res["processed"] = True
                                break
                else: # Single item group
                    for res in analysis_results:
                        if res["suggested_name"] == original_list[0] and not res["processed"]:
                            _, ext = os.path.splitext(original_list[0])
                            new_grouped_name = f"{safe_uniform_base}{ext}"
                            final_rename_plan[res["original_path"]] = new_grouped_name
                            if original_list[0] != new_grouped_name:
                                log_message(f"  Grouping '{original_list[0]}' -> '{new_grouped_name}' (uniform base applied)", log_callback)
                            res["processed"] = True
                            break
        except Exception as e:
            log_message(f"  Error during grouping: {e}", log_callback)
            if 'response' in locals() and 'response' in response: log_message(f"  Raw LLM response was:\n---\n{response['response']}\n---", log_callback)
            log_message("  Skipping grouping and proceeding with original suggestions.", log_callback)

    # --- FIX: Ensure all files have a plan, even if they were missed by the grouping LLM ---
    for res in analysis_results:
        if res["original_path"] not in final_rename_plan:
            log_message(f"  Info: '{res['original_name']}' was not grouped. Using its original suggestion.", log_callback)
            final_rename_plan[res["original_path"]] = res["suggested_name"]
    
    # PASS 3: RENAMING
    log_message("\n--- Starting Pass 3: Finalizing and Renaming Files ---", log_callback)
    files_renamed = 0
    for res in analysis_results:
        if stop_flag_check(): break
        filepath = res["original_path"]
        original_filename = res["original_name"]
        new_name_from_plan = final_rename_plan.get(filepath)
        
        if not new_name_from_plan:
            log_message(f"Could not find a rename plan for {original_filename}. Skipping.", log_callback)
            continue
        
        final_name_to_use = new_name_from_plan
        if gui_opts["keep_original_name"]:
            original_base, original_ext = os.path.splitext(original_filename)
            new_base, _ = os.path.splitext(new_name_from_plan)
            final_name_to_use = f"{original_base}_{new_base}{original_ext}"
        
        if final_name_to_use.lower() == original_filename.lower():
            continue

        counter = 1
        final_new_filepath = os.path.join(target_folder, final_name_to_use)
        while os.path.exists(final_new_filepath) and final_new_filepath.lower() != filepath.lower():
            base, ext = os.path.splitext(final_name_to_use)
            final_name_to_use = f"{base}_{counter}{ext}"
            final_new_filepath = os.path.join(target_folder, final_name_to_use)
            counter += 1
        
        log_message(f"\nOriginal:  {original_filename}\nSuggested: {final_name_to_use}", log_callback)
        if gui_opts["dry_run"]:
            log_message("  DRY RUN: No rename performed.", log_callback)
        elif not gui_opts["auto_confirm"]:
            log_message("  Rename skipped (auto-confirm not checked).", log_callback)
        else:
            try:
                os.rename(filepath, final_new_filepath)
                log_message(f"  SUCCESS: Renamed to '{final_name_to_use}'", log_callback)
                files_renamed += 1
            except Exception as e:
                log_message(f"  ERROR renaming: {e}", log_callback)

    log_message(f"\nRenaming complete. Renamed {files_renamed} of {len(analysis_results)} analyzed files.", log_callback)

def setup_mimetypes():
    """Adds custom mime types to the system's database."""
    types = {"video/mts": ".mts", "application/pdf": ".pdf"}
    for mime, ext in types.items():
        mimetypes.add_type(mime, ext, strict=False)