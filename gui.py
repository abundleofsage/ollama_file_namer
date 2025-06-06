# gui.py
import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import sys
import threading
import traceback
import ollama
import mimetypes
import queue
import os

# Import from our new modules
import config as cfg
import renamer_core as core

# --- Global State for GUI ---
is_processing_running = False
stop_processing_flag = False

class TextRedirector:
    """Redirects print statements to a tkinter widget."""
    def __init__(self, widget, queue):
        self.widget = widget
        self.queue = queue
    def write(self, s):
        self.queue.put(s)
    def flush(self):
        pass

class PromptEditorWindow(tk.Toplevel):
    """A Toplevel window for editing prompts."""
    def __init__(self, parent, config):
        super().__init__(parent)
        self.config = config
        self.transient(parent)
        self.title("Prompt Editor")
        self.geometry("700x550")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(pady=10, padx=10, expand=True, fill="both")
        self.prompt_texts = {}
        prompt_keys = list(cfg.DEFAULT_PROMPTS.keys())
        for key in prompt_keys:
            frame = ttk.Frame(self.notebook, padding="10")
            self.notebook.add(frame, text=key.replace('_', ' ').title())
            text_widget = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=80, height=20)
            text_widget.pack(expand=True, fill="both")
            text_widget.insert(tk.END, self.config.get("prompts", {}).get(key, ""))
            self.prompt_texts[key] = text_widget
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=5, padx=10, fill='x')
        ttk.Button(button_frame, text="Save & Close", command=self.save_and_close).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=10)

    def save_and_close(self):
        for key, text_widget in self.prompt_texts.items():
            self.config["prompts"][key] = text_widget.get("1.0", tk.END).strip()
        cfg.save_config(self.config)
        self.destroy()

class RenamerApp:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.root.title("Ollama File Renamer")
        self.root.geometry("800x650")

        # --- Variables ---
        self.folder_path = tk.StringVar()
        self.text_model_var = tk.StringVar()
        self.vision_model_var = tk.StringVar()
        self.temperature_var = tk.DoubleVar(value=self.config["generation_parameters"]["temperature"])
        self.video_frames_var = tk.IntVar(value=self.config["video_processing"]["frames_to_analyze"])
        self.dry_run_var = tk.BooleanVar(value=True)
        self.keep_original_name_var = tk.BooleanVar(value=False)
        self.skip_metadata_var = tk.BooleanVar(value=False)
        self.auto_confirm_var = tk.BooleanVar(value=False)
        self.group_similar_var = tk.BooleanVar(value=False)

        # --- GUI Setup ---
        self.setup_ui()
        
        # --- Redirector Setup ---
        self.log_queue = queue.SimpleQueue()
        sys.stdout = TextRedirector(self.console, self.log_queue)
        sys.stderr = TextRedirector(self.console, self.log_queue)
        
        self.update_log_widget()
        self.populate_model_lists()
        
        print(f"Ollama File Renamer GUI started. Loaded config from {cfg.DEFAULT_CONFIG_FILENAME}")
        if not core.PYPDF_AVAILABLE:
            print("Warning: pypdf not found. PDF text extraction is disabled. Run: pip install pypdf")

    def update_log_widget(self):
        """Checks the queue for new log messages and updates the widget."""
        while not self.log_queue.empty():
            message = self.log_queue.get()
            self.console.configure(state='normal')
            self.console.insert(tk.END, message)
            self.console.see(tk.END)
            self.console.configure(state='disabled')
        self.root.after(100, self.update_log_widget)
        
    def setup_ui(self):
        """Creates and places all the widgets in the main window."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Folder Selection
        folder_frame = ttk.LabelFrame(main_frame, text="1. Select Folder", padding="10")
        folder_frame.pack(fill=tk.X, pady=5)
        ttk.Entry(folder_frame, textvariable=self.folder_path, width=70).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(folder_frame, text="Browse...", command=self.browse_folder).pack(side=tk.LEFT)
        
        # Model Selection
        model_frame = ttk.LabelFrame(main_frame, text="2. Select Models", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="Vision Model:").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        self.vision_model_combo = ttk.Combobox(model_frame, textvariable=self.vision_model_var, width=40)
        self.vision_model_combo.grid(row=0, column=1, padx=5, pady=2, sticky='ew')
        ttk.Label(model_frame, text="Text Model:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        self.text_model_combo = ttk.Combobox(model_frame, textvariable=self.text_model_var, width=40)
        self.text_model_combo.grid(row=1, column=1, padx=5, pady=2, sticky='ew')
        ttk.Button(model_frame, text="Refresh Models", command=self.populate_model_lists).grid(row=0, column=2, rowspan=2, padx=10)
        model_frame.columnconfigure(1, weight=1)

        # Options
        options_frame = ttk.LabelFrame(main_frame, text="3. Set Options", padding="10")
        options_frame.pack(fill=tk.X, pady=5)
        left_options = ttk.Frame(options_frame); left_options.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Checkbutton(left_options, text="Dry Run (Preview changes)", variable=self.dry_run_var).pack(anchor='w')
        ttk.Checkbutton(left_options, text="Keep Original Name (Append)", variable=self.keep_original_name_var).pack(anchor='w')
        ttk.Checkbutton(left_options, text="Skip Metadata in Prompts", variable=self.skip_metadata_var).pack(anchor='w')
        ttk.Checkbutton(left_options, text="Group Similar Filenames (Slower)", variable=self.group_similar_var).pack(anchor='w')
        ttk.Checkbutton(left_options, text="Auto-confirm all (USE WITH CAUTION)", variable=self.auto_confirm_var).pack(anchor='w')
        
        right_options = ttk.Frame(options_frame); right_options.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(right_options, text="Temperature:").pack(anchor='w')
        ttk.Scale(right_options, from_=0.0, to=1.0, orient='horizontal', variable=self.temperature_var).pack(anchor='w', fill=tk.X)
        ttk.Label(right_options, text="Video Frames to Analyze:").pack(anchor='w', pady=(5,0))
        ttk.Spinbox(right_options, from_=0, to=50, textvariable=self.video_frames_var, width=5).pack(anchor='w')

        # Console
        console_frame = ttk.LabelFrame(main_frame, text="4. Log Output", padding="10")
        console_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.console = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, state='disabled', height=10)
        self.console.pack(fill=tk.BOTH, expand=True)

        # Action Buttons
        action_frame = ttk.Frame(main_frame, padding=(0, 10, 0, 0))
        action_frame.pack(fill=tk.X)
        ttk.Button(action_frame, text="Edit Prompts...", command=self.open_prompt_editor).pack(side=tk.LEFT)
        self.run_button = ttk.Button(action_frame, text="Run Renamer", command=self.start_processing_thread)
        self.run_button.pack(side=tk.RIGHT, padx=5)
        self.stop_button = ttk.Button(action_frame, text="Stop", command=self.stop_processing, state='disabled')
        self.stop_button.pack(side=tk.RIGHT)

    def open_prompt_editor(self):
        PromptEditorWindow(self.root, self.config)

    def populate_model_lists(self):
        print("Fetching Ollama models...")
        model_names = []  # Initialize empty list
        try:
            client = ollama.Client(host=self.config.get("ollama_host"))
            models_info = client.list()
            
            # Check if the 'models' key exists and is a list
            if 'models' in models_info and isinstance(models_info['models'], list):
                for model_data in models_info['models']:
                    # Based on your initial log, the key is 'model'. 
                    # We will also check for 'name' as a fallback.
                    if 'model' in model_data:
                        model_names.append(model_data['model'])
                    elif 'name' in model_data:
                        model_names.append(model_data['name'])
                    else:
                        print(f"Warning: Found a model entry with no 'model' or 'name' key: {model_data}")

            # Set the values for the comboboxes
            self.vision_model_combo['values'] = model_names
            self.text_model_combo['values'] = model_names
            
            if not model_names:
                print("Warning: No models were found. The dropdowns will be empty.")
            
            # Set default models from config
            vision_model = self.config.get("models", {}).get("vision_model")
            if vision_model in model_names:
                self.vision_model_var.set(vision_model)

            text_model = self.config.get("models", {}).get("text_model")
            if text_model in model_names:
                self.text_model_var.set(text_model)
                
            print("Model lists updated.")
        except Exception as e:
            print(f"ERROR: An exception occurred while fetching models: {e}")
            messagebox.showerror("Ollama Error", f"Could not fetch models from Ollama.\nError: {e}\n\nMake sure Ollama is running.")

    def browse_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.folder_path.set(path)
            print(f"Selected folder: {path}")

    def start_processing_thread(self):
        global is_processing_running, stop_processing_flag
        if is_processing_running:
            messagebox.showwarning("Busy", "A renaming process is already running.")
            return
            
        target_folder = self.folder_path.get()
        if not target_folder or not os.path.isdir(target_folder):
            messagebox.showerror("Error", "Please select a valid folder.")
            return
            
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')
        is_processing_running = True
        stop_processing_flag = False
        
        self.console.configure(state='normal')
        self.console.delete('1.0', tk.END)
        self.console.configure(state='disabled')

        # Gather options for the core logic
        gui_opts = {
            "dry_run": self.dry_run_var.get(),
            "keep_original_name": self.keep_original_name_var.get(),
            "skip_metadata": self.skip_metadata_var.get(),
            "group_similar": self.group_similar_var.get(),
            "auto_confirm": self.auto_confirm_var.get()
        }
        
        # Update config with current GUI settings before starting
        self.config["models"]["text_model"] = self.text_model_var.get()
        self.config["models"]["vision_model"] = self.vision_model_var.get()
        self.config["generation_parameters"]["temperature"] = self.temperature_var.get()
        self.config["video_processing"]["frames_to_analyze"] = self.video_frames_var.get()

        thread = threading.Thread(
            target=self.process_folder_wrapper,
            args=(target_folder, gui_opts, self.config)
        )
        thread.daemon = True
        thread.start()

    def stop_processing(self):
        global stop_processing_flag
        if is_processing_running:
            print("\n>>> STOP signal received. <<<\n")
            stop_processing_flag = True
            self.stop_button.config(state='disabled')

    def process_folder_wrapper(self, target_folder, gui_opts, current_config):
        """Wrapper to run the core logic and handle final state."""
        try:
            core.process_directory(
                target_folder,
                gui_opts,
                current_config,
                stop_flag_check=lambda: stop_processing_flag,
                log_callback=lambda msg: self.log_queue.put(msg)
            )
        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = (
                f"\n--- UNHANDLED FATAL ERROR in processing thread ---\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Details: {e}\n"
                f"Traceback:\n{tb_str}\n"
                f"--- END ERROR ---\n"
            )
            self.log_queue.put(error_msg)
        finally:
            self.root.after(0, self.on_processing_finished)

    def on_processing_finished(self):
        global is_processing_running
        is_processing_running = False
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        print("\n>>> Processing finished. <<<")

if __name__ == "__main__":
    core.setup_mimetypes()
    app_config = cfg.load_config()
    root = tk.Tk()
    app = RenamerApp(root, app_config)
    root.mainloop()