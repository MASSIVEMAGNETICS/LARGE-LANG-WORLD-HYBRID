"""Main GUI Application - ChatGPT-style interface for the Hybrid AI."""

try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox, filedialog
except ImportError:
    import Tkinter as tk
    import ttk
    import ScrolledText as scrolledtext
    import tkMessageBox as messagebox
    import tkFileDialog as filedialog

import sys
import os
import threading
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hybrid_model import HybridLanguageWorldModel
from training.trainer import ModelTrainer
from agents.pipeline_builder import PipelineBuilder


class MainApplication(tk.Tk):
    """
    Main GUI Application providing ChatGPT-style interface.
    Compatible with Windows 7 and uses tkinter (built-in).
    """
    
    def __init__(self):
        """Initialize the main application."""
        tk.Tk.__init__(self)
        
        self.title("Large Language-World Hybrid AI - REVOLUTIONARY AI SYSTEM")
        self.geometry("1200x800")
        
        # Initialize model (lazy loading)
        self.model = None
        self.trainer = None
        self.pipeline_builder = None
        
        # Chat history
        self.chat_history = []
        
        # AI-to-AI chat models
        self.ai_models = {}
        
        # Setup UI
        self.setup_menu()
        self.setup_notebook()
        
        # Status bar
        self.status_bar = tk.Label(self, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.update_status("Application started - Load or create a model to begin")
    
    def setup_menu(self):
        """Setup the menu bar."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Model", command=self.new_model)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_command(label="Save Model", command=self.save_model)
        file_menu.add_command(label="Export Model", command=self.export_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_notebook(self):
        """Setup the tabbed notebook interface."""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: Chat Interface
        self.setup_chat_tab()
        
        # Tab 2: Training Suite
        self.setup_training_tab()
        
        # Tab 3: Pipeline Builder
        self.setup_pipeline_tab()
        
        # Tab 4: AI-to-AI Chat
        self.setup_ai_to_ai_tab()
    
    def setup_chat_tab(self):
        """Setup the main chat interface tab."""
        chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(chat_frame, text="Chat Interface")
        
        # Chat display area
        chat_display_frame = ttk.Frame(chat_frame)
        chat_display_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(chat_display_frame, text="Chat with Hybrid AI:", 
                 font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_display_frame,
            wrap=tk.WORD,
            width=80,
            height=25,
            font=("Arial", 10)
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=5)
        self.chat_display.config(state=tk.DISABLED)
        
        # Input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Your message:").pack(anchor=tk.W)
        
        self.chat_input = tk.Text(input_frame, height=3, font=("Arial", 10))
        self.chat_input.pack(fill=tk.X, pady=2)
        self.chat_input.bind("<Control-Return>", lambda e: self.send_message())
        
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X)
        
        ttk.Button(button_frame, text="Send (Ctrl+Enter)", 
                  command=self.send_message).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clear Chat", 
                  command=self.clear_chat).pack(side=tk.LEFT, padx=2)
        
        # Temperature control
        temp_frame = ttk.Frame(button_frame)
        temp_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Label(temp_frame, text="Temperature:").pack(side=tk.LEFT)
        self.temperature_var = tk.DoubleVar(value=0.7)
        self.temperature_scale = ttk.Scale(
            temp_frame,
            from_=0.1,
            to=2.0,
            variable=self.temperature_var,
            orient=tk.HORIZONTAL,
            length=150
        )
        self.temperature_scale.pack(side=tk.LEFT, padx=5)
        self.temp_label = ttk.Label(temp_frame, text="0.7")
        self.temp_label.pack(side=tk.LEFT)
        self.temperature_var.trace("w", self.update_temp_label)
    
    def setup_training_tab(self):
        """Setup the training suite tab."""
        train_frame = ttk.Frame(self.notebook)
        self.notebook.add(train_frame, text="Training Suite")
        
        # Training controls
        control_frame = ttk.LabelFrame(train_frame, text="Training Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Dataset selection
        ttk.Label(control_frame, text="Training Dataset:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.dataset_var = tk.StringVar()
        dataset_entry = ttk.Entry(control_frame, textvariable=self.dataset_var, width=50)
        dataset_entry.grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(control_frame, text="Browse", 
                  command=self.browse_dataset).grid(row=0, column=2, padx=5, pady=2)
        
        # Epochs
        ttk.Label(control_frame, text="Epochs:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.IntVar(value=10)
        ttk.Spinbox(control_frame, from_=1, to=1000, textvariable=self.epochs_var,
                   width=20).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Batch size
        ttk.Label(control_frame, text="Batch Size:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Spinbox(control_frame, from_=1, to=256, textvariable=self.batch_size_var,
                   width=20).grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Learning rate
        ttk.Label(control_frame, text="Learning Rate:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.DoubleVar(value=0.001)
        ttk.Entry(control_frame, textvariable=self.lr_var,
                 width=20).grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        self.train_button = ttk.Button(button_frame, text="Start Training", 
                                       command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Training",
                                      command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Training log
        log_frame = ttk.LabelFrame(train_frame, text="Training Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20)
        self.training_log.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(train_frame, variable=self.progress_var, 
                                           maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
    
    def setup_pipeline_tab(self):
        """Setup the pipeline action agent builder tab."""
        pipeline_frame = ttk.Frame(self.notebook)
        self.notebook.add(pipeline_frame, text="Pipeline Builder")
        
        # Split into two panes
        paned = ttk.PanedWindow(pipeline_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left: Agent blocks
        left_frame = ttk.LabelFrame(paned, text="Available Agent Blocks", padding=10)
        paned.add(left_frame, weight=1)
        
        ttk.Label(left_frame, text="Drag blocks to create pipeline:", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.agent_blocks = tk.Listbox(left_frame, height=20)
        self.agent_blocks.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add sample agent blocks
        agent_types = [
            "Text Input Agent",
            "Language Processing Agent",
            "World State Agent",
            "Reasoning Agent",
            "Action Agent",
            "Output Agent",
            "Conditional Branch",
            "Loop Agent",
            "API Call Agent",
            "File I/O Agent"
        ]
        for agent in agent_types:
            self.agent_blocks.insert(tk.END, agent)
        
        # Right: Pipeline canvas
        right_frame = ttk.LabelFrame(paned, text="Pipeline Design", padding=10)
        paned.add(right_frame, weight=2)
        
        # Pipeline canvas
        self.pipeline_canvas = tk.Canvas(right_frame, bg="white", height=400)
        self.pipeline_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Pipeline controls
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Add Block", 
                  command=self.add_pipeline_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Clear Pipeline",
                  command=self.clear_pipeline).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Save Pipeline",
                  command=self.save_pipeline).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Load Pipeline",
                  command=self.load_pipeline).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="Run Pipeline",
                  command=self.run_pipeline).pack(side=tk.LEFT, padx=2)
        
        # Pipeline output
        output_frame = ttk.LabelFrame(right_frame, text="Pipeline Output", padding=5)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.pipeline_output = scrolledtext.ScrolledText(output_frame, height=8, wrap=tk.WORD)
        self.pipeline_output.pack(fill=tk.BOTH, expand=True)
    
    def setup_ai_to_ai_tab(self):
        """Setup the AI-to-AI chat tab."""
        ai_chat_frame = ttk.Frame(self.notebook)
        self.notebook.add(ai_chat_frame, text="AI-to-AI Chat")
        
        # Top: Model selection
        model_frame = ttk.LabelFrame(ai_chat_frame, text="AI Models", padding=10)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_frame, text="Select AI models to interact:").pack(anchor=tk.W)
        
        models_grid = ttk.Frame(model_frame)
        models_grid.pack(fill=tk.X, pady=5)
        
        ttk.Label(models_grid, text="Model 1:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.model1_var = tk.StringVar(value="Hybrid AI (Primary)")
        self.model1_combo = ttk.Combobox(models_grid, textvariable=self.model1_var,
                                         values=["Hybrid AI (Primary)", "Language Model Only",
                                                "World Model Only"], width=30)
        self.model1_combo.grid(row=0, column=1, padx=5)
        
        ttk.Label(models_grid, text="Model 2:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.model2_var = tk.StringVar(value="Language Model Only")
        self.model2_combo = ttk.Combobox(models_grid, textvariable=self.model2_var,
                                         values=["Hybrid AI (Primary)", "Language Model Only",
                                                "World Model Only"], width=30)
        self.model2_combo.grid(row=0, column=3, padx=5)
        
        ttk.Button(model_frame, text="Start Conversation",
                  command=self.start_ai_conversation).pack(pady=5)
        
        # Middle: Conversation display
        conv_frame = ttk.LabelFrame(ai_chat_frame, text="AI-to-AI Conversation", padding=10)
        conv_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ai_conversation = scrolledtext.ScrolledText(conv_frame, wrap=tk.WORD, height=20)
        self.ai_conversation.pack(fill=tk.BOTH, expand=True)
        self.ai_conversation.config(state=tk.DISABLED)
        
        # Bottom: Controls
        control_frame = ttk.Frame(ai_chat_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Topic/Prompt:").pack(side=tk.LEFT, padx=5)
        self.ai_topic_var = tk.StringVar()
        ttk.Entry(control_frame, textvariable=self.ai_topic_var, 
                 width=50).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(control_frame, text="Turns:").pack(side=tk.LEFT, padx=5)
        self.turns_var = tk.IntVar(value=5)
        ttk.Spinbox(control_frame, from_=1, to=50, textvariable=self.turns_var,
                   width=10).pack(side=tk.LEFT, padx=5)
    
    # Chat Interface Methods
    def send_message(self):
        """Send a message to the AI."""
        message = self.chat_input.get("1.0", tk.END).strip()
        if not message:
            return
        
        if self.model is None:
            messagebox.showwarning("No Model", "Please load or create a model first!")
            return
        
        # Clear input
        self.chat_input.delete("1.0", tk.END)
        
        # Display user message
        self.display_chat_message("You", message)
        
        # Get AI response (in separate thread to avoid blocking)
        threading.Thread(target=self.get_ai_response, args=(message,), daemon=True).start()
        
        self.update_status("Generating response...")
    
    def get_ai_response(self, message):
        """Get AI response to message."""
        try:
            # Simulate processing (replace with actual model inference)
            import time
            time.sleep(1)
            
            response = "AI Response: I am the Large Language-World Hybrid AI. " \
                      "I combine language understanding with world modeling for superior reasoning. " \
                      "Your message: '{}'".format(message)
            
            # Display AI response
            self.after(0, self.display_chat_message, "Hybrid AI", response)
            self.after(0, self.update_status, "Ready")
        except Exception as e:
            self.after(0, messagebox.showerror, "Error", str(e))
            self.after(0, self.update_status, "Error generating response")
    
    def display_chat_message(self, sender, message):
        """Display a chat message."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, "{}: {}\n\n".format(sender, message))
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        self.chat_history.append({"sender": sender, "message": message})
    
    def clear_chat(self):
        """Clear the chat display."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_history = []
    
    def update_temp_label(self, *args):
        """Update temperature label."""
        self.temp_label.config(text="{:.2f}".format(self.temperature_var.get()))
    
    # Model Management Methods
    def new_model(self):
        """Create a new model."""
        self.model = HybridLanguageWorldModel()
        self.trainer = ModelTrainer(self.model)
        messagebox.showinfo("Success", "New model created successfully!")
        self.update_status("New model created")
    
    def load_model(self):
        """Load a model from file."""
        filepath = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("PyTorch Models", "*.pt *.pth"), ("All Files", "*.*")]
        )
        if filepath:
            try:
                self.model = HybridLanguageWorldModel.load_model(filepath)
                self.trainer = ModelTrainer(self.model)
                messagebox.showinfo("Success", "Model loaded successfully!")
                self.update_status("Model loaded from: {}".format(filepath))
            except Exception as e:
                messagebox.showerror("Error", "Failed to load model: {}".format(str(e)))
    
    def save_model(self):
        """Save the current model."""
        if self.model is None:
            messagebox.showwarning("No Model", "No model to save!")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pt",
            filetypes=[("PyTorch Models", "*.pt"), ("All Files", "*.*")]
        )
        if filepath:
            try:
                self.model.save_model(filepath)
                messagebox.showinfo("Success", "Model saved successfully!")
                self.update_status("Model saved to: {}".format(filepath))
            except Exception as e:
                messagebox.showerror("Error", "Failed to save model: {}".format(str(e)))
    
    def export_model(self):
        """Export model to ONNX."""
        if self.model is None:
            messagebox.showwarning("No Model", "No model to export!")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Export Model",
            defaultextension=".onnx",
            filetypes=[("ONNX Models", "*.onnx"), ("All Files", "*.*")]
        )
        if filepath:
            try:
                import torch
                sample_input = torch.randint(0, 100, (1, 10))
                self.model.export_to_onnx(filepath, sample_input)
                messagebox.showinfo("Success", "Model exported to ONNX successfully!")
                self.update_status("Model exported to: {}".format(filepath))
            except Exception as e:
                messagebox.showerror("Error", "Failed to export model: {}".format(str(e)))
    
    # Training Methods
    def browse_dataset(self):
        """Browse for dataset."""
        filepath = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("Text Files", "*.txt"), ("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filepath:
            self.dataset_var.set(filepath)
    
    def start_training(self):
        """Start model training."""
        if self.model is None:
            messagebox.showwarning("No Model", "Please create or load a model first!")
            return
        
        dataset_path = self.dataset_var.get()
        if not dataset_path:
            messagebox.showwarning("No Dataset", "Please select a training dataset!")
            return
        
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start training in separate thread
        threading.Thread(target=self.train_model, daemon=True).start()
    
    def train_model(self):
        """Train the model."""
        try:
            epochs = self.epochs_var.get()
            batch_size = self.batch_size_var.get()
            lr = self.lr_var.get()
            
            self.log_training("Starting training...")
            self.log_training("Epochs: {}, Batch Size: {}, LR: {}".format(epochs, batch_size, lr))
            
            # Simulate training
            import time
            for epoch in range(epochs):
                self.log_training("Epoch {}/{}".format(epoch + 1, epochs))
                time.sleep(0.5)
                progress = ((epoch + 1) / float(epochs)) * 100
                self.after(0, self.progress_var.set, progress)
            
            self.log_training("Training completed!")
            self.after(0, messagebox.showinfo, "Success", "Training completed!")
        except Exception as e:
            self.log_training("Error: {}".format(str(e)))
            self.after(0, messagebox.showerror, "Error", str(e))
        finally:
            self.after(0, self.train_button.config, {"state": tk.NORMAL})
            self.after(0, self.stop_button.config, {"state": tk.DISABLED})
    
    def stop_training(self):
        """Stop training."""
        self.log_training("Training stopped by user")
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def log_training(self, message):
        """Log training message."""
        self.after(0, self._log_training_safe, message)
    
    def _log_training_safe(self, message):
        """Safely log training message from any thread."""
        self.training_log.insert(tk.END, message + "\n")
        self.training_log.see(tk.END)
    
    # Pipeline Methods
    def add_pipeline_block(self):
        """Add a block to the pipeline."""
        selection = self.agent_blocks.curselection()
        if not selection:
            messagebox.showinfo("No Selection", "Please select an agent block to add!")
            return
        
        block_name = self.agent_blocks.get(selection[0])
        self.pipeline_output.insert(tk.END, "Added block: {}\n".format(block_name))
        
        # Draw on canvas (simple representation)
        import random
        x, y = random.randint(50, 400), random.randint(50, 300)
        self.pipeline_canvas.create_rectangle(x, y, x+100, y+50, fill="lightblue", outline="blue")
        self.pipeline_canvas.create_text(x+50, y+25, text=block_name.split()[0])
    
    def clear_pipeline(self):
        """Clear the pipeline."""
        self.pipeline_canvas.delete("all")
        self.pipeline_output.delete("1.0", tk.END)
    
    def save_pipeline(self):
        """Save pipeline configuration."""
        filepath = filedialog.asksaveasfilename(
            title="Save Pipeline",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filepath:
            # Save pipeline data
            pipeline_data = {"blocks": [], "connections": []}
            with open(filepath, 'w') as f:
                json.dump(pipeline_data, f, indent=2)
            messagebox.showinfo("Success", "Pipeline saved!")
    
    def load_pipeline(self):
        """Load pipeline configuration."""
        filepath = filedialog.askopenfilename(
            title="Load Pipeline",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    pipeline_data = json.load(f)
                messagebox.showinfo("Success", "Pipeline loaded!")
                self.pipeline_output.insert(tk.END, "Loaded pipeline from: {}\n".format(filepath))
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def run_pipeline(self):
        """Run the pipeline."""
        self.pipeline_output.insert(tk.END, "\n=== Running Pipeline ===\n")
        self.pipeline_output.insert(tk.END, "Pipeline execution started...\n")
        # Actual execution would happen here
        self.pipeline_output.insert(tk.END, "Pipeline completed!\n")
    
    # AI-to-AI Chat Methods
    def start_ai_conversation(self):
        """Start AI-to-AI conversation."""
        topic = self.ai_topic_var.get()
        if not topic:
            messagebox.showwarning("No Topic", "Please enter a topic for the conversation!")
            return
        
        turns = self.turns_var.get()
        model1 = self.model1_var.get()
        model2 = self.model2_var.get()
        
        self.ai_conversation.config(state=tk.NORMAL)
        self.ai_conversation.delete("1.0", tk.END)
        self.ai_conversation.insert(tk.END, "=== AI-to-AI Conversation ===\n")
        self.ai_conversation.insert(tk.END, "Topic: {}\n".format(topic))
        self.ai_conversation.insert(tk.END, "Model 1: {}\n".format(model1))
        self.ai_conversation.insert(tk.END, "Model 2: {}\n\n".format(model2))
        
        # Simulate conversation
        for i in range(turns):
            self.ai_conversation.insert(tk.END, "{}: Response about '{}' turn {}\n\n".format(
                model1, topic, i+1))
            self.ai_conversation.insert(tk.END, "{}: Counter-response turn {}\n\n".format(
                model2, i+1))
        
        self.ai_conversation.insert(tk.END, "\n=== Conversation Complete ===\n")
        self.ai_conversation.see(tk.END)
        self.ai_conversation.config(state=tk.DISABLED)
    
    # Utility Methods
    def update_status(self, message):
        """Update status bar."""
        self.status_bar.config(text=message)
    
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About",
            "Large Language-World Hybrid AI\n\n"
            "A revolutionary AI system combining language and world models\n\n"
            "Version 1.0.0\n"
            "By MASSIVE MAGNETICS"
        )


def main():
    """Main entry point for the GUI application."""
    app = MainApplication()
    app.mainloop()


if __name__ == "__main__":
    main()
