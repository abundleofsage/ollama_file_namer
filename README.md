# **Ollama File Namer**

This Python script analyzes files in a specified folder using Ollama with local large language models (LLMs) to generate new, descriptive filenames based on their content. It supports text files, images, PDFs (text extraction), and video files (by analyzing extracted frames).

## **Features**

* Analyzes various file types:  
  * **Images:** Uses a vision model (e.g., LLaVA) to describe image content.  
  * **Videos:** Extracts frames, describes them with a vision model, and then summarizes these descriptions with a text model.  
  * **PDFs:** Extracts text content (for text-based PDFs) using pypdf and analyzes it with a text model.  
  * **Text Files:** Analyzes plain text content with a text model.  
* Generates descriptive filenames based on content and metadata.  
* Option to keep the original filename and append the new suggestion.  
* Customizable Ollama models for text and vision.  
* Adjustable LLM generation parameters (temperature, number of tokens to predict).  
* Dry-run mode to preview changes before renaming.  
* Option to skip files by extension.  
* User confirmation for each rename (can be overridden).

## **Dependencies**

1. **Python 3.x**  
2. **Ollama:** Ensure Ollama is installed and running. You can download it from [ollama.com](https://ollama.com).  
   * You'll also need to pull the models you intend to use via the Ollama CLI. For example:  
     ollama pull llava:latest  \# Default vision model  
     ollama pull gemma:2b     \# Default text model

3. **FFmpeg:** Required for video file processing (frame extraction and duration). It must be installed and accessible in your system's PATH.  
   * **Linux (Debian/Ubuntu):** sudo apt update && sudo apt install ffmpeg  
   * **macOS (Homebrew):** brew install ffmpeg  
   * **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add the bin folder to your PATH.  
   * Verify by running ffmpeg \-version and ffprobe \-version in your terminal.  
4. **Python Libraries:**  
   * ollama: The official Python client for Ollama.  
     pip install ollama

   * pypdf: For extracting text from PDF files.  
     pip install pypdf

## **Usage**

Save the main.py script and make it executable if necessary (chmod \+x ollama\_rename.py on Linux/macOS).

python main.py \<folder\_path\> \[options\]

### **Arguments**

* folder: (Required) Path to the folder containing files to rename.

### **Options**

* \--text-model TEXT\_MODEL: Ollama model for text files (default: gemma:2b). Set to an empty string ("") to skip text file processing.  
* \--vision-model VISION\_MODEL: Ollama model for image and video frame analysis (default: llava:latest). Set to an empty string ("") to skip image/video frame processing.  
* \--ollama-host OLLAMA\_HOST: Ollama server address (default: http://localhost:11434).  
* \--dry-run: Show proposed renames without actually renaming files. (Highly recommended for first runs).  
* \--skip-extensions .ext1 .ext2 ...: List of file extensions to skip (e.g., .log .tmp). Remember to include the leading dot.  
* \--max-text-chars NUM: Maximum number of characters to send from text files (including PDFs) for analysis (default: 2000).  
* \--temperature TEMP: Temperature for Ollama model generation (a float, default: 0.4). Higher values (e.g., 0.7) make output more random; lower values (e.g., 0.2) make it more deterministic.  
* \--num-predict NUM\_PREDICT: Maximum number of tokens for the LLM to predict (default: 70). Affects the length of the generated description/filename.  
* \--video-frames NUM\_FRAMES: Number of frames to analyze from video files (default: 5). Set to 0 to disable video processing.  
* \--keep-original-name: If specified, the LLM-suggested name (base) will be appended to the original filename (base), keeping the original extension. E.g., original.jpg \+ new\_suggestion \-\> original\_new\_suggestion.jpg.  
* \-y, \--yes: Automatically confirm all renames. **Use with extreme caution\!**

## **Examples**

1. **Dry run on a folder named "MyPhotos", using default models:**  
   python ollama\_rename.py ./MyPhotos \--dry-run

2. **Process** files **in "Documents", using mistral for text and skipping images, auto-confirming renames (caution\!):**  
   python ollama\_rename.py /path/to/Documents \--text-model mistral \--vision-model "" \-y

3. **Process videos in "VacationClips", analyzing 10 frames per video, and keeping the original name part:**  
   python ollama\_rename.py ./VacationClips \--video-frames 10 \--keep-original-name

4. **Process images in "Artwork" with a higher temperature for more varied suggestions:**  
   python ollama\_rename.py ./Artwork \--temperature 0.7 \--text-model "" 

5. **Process PDFs in "Reports", skipping .bak files:**  
   python ollama\_rename.py ./Reports \--skip-extensions .bak \--vision-model "" \--video-frames 0

## **Notes**

* **Backup your files\!** Especially before running without \--dry-run or with \-y.  
* Processing can be slow, especially for videos, as it involves multiple LLM calls and FFmpeg operations per file.  
* The quality of filenames depends heavily on the chosen LLMs and the clarity of the file content.  
* Ensure FFmpeg is correctly installed and in your system's PATH if you want to process video files.  
* If pypdf is not installed, PDF text extraction will be skipped, and the script will rely on metadata for PDFs.
