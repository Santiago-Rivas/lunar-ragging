import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class PlotViewer:
    def __init__(self, root, plot_dir):
        self.root = root
        self.root.title("Metrics Plots Viewer")
        self.root.geometry("1200x800")
        
        # List of plot files with descriptive names
        self.plot_files = sorted([f for f in os.listdir(plot_dir) if f.endswith('.png')])
        self.plot_dir = plot_dir
        self.current_index = 0
        
        # Create descriptive names for the plots
        self.plot_descriptions = {
            "3d_parameter_space.png": "3D visualization of chunk_size, chunk_overlap, and k vs metrics",
            "chunk_overlap_effect.png": "Effect of chunk overlap on metrics across different k values",
            "chunk_overlap_vs_metrics.png": "Relationship between chunk overlap and metrics",
            "chunk_size_effect.png": "Effect of chunk size on metrics across different k values",
            "chunk_size_vs_metrics.png": "Relationship between chunk size and metrics",
            "heatmap_metrics.png": "Heatmap of metrics by chunk size and overlap",
            "k_vs_metrics.png": "Relationship between k (number of retrieved chunks) and metrics",
            "user_patterns.png": "User-specific patterns in metrics"
        }
        
        # Main frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image display
        self.img_label = ttk.Label(main_frame)
        self.img_label.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Description label
        self.desc_label = ttk.Label(main_frame, font=("Arial", 12), wraplength=1100)
        self.desc_label.pack(fill=tk.X, pady=10)
        
        # Navigation buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        # Previous button
        prev_btn = ttk.Button(btn_frame, text="Previous", command=self.show_previous)
        prev_btn.pack(side=tk.LEFT, padx=5)
        
        # Next button
        next_btn = ttk.Button(btn_frame, text="Next", command=self.show_next)
        next_btn.pack(side=tk.RIGHT, padx=5)
        
        # Center label showing current plot
        self.counter_label = ttk.Label(btn_frame, font=("Arial", 10))
        self.counter_label.pack(side=tk.TOP, padx=5)
        
        # Dropdown to select plot
        self.plot_var = tk.StringVar()
        plot_dropdown = ttk.Combobox(main_frame, textvariable=self.plot_var, state="readonly")
        plot_dropdown['values'] = [f"{i+1}. {f}" for i, f in enumerate(self.plot_files)]
        plot_dropdown.bind('<<ComboboxSelected>>', self.on_plot_selected)
        plot_dropdown.pack(fill=tk.X, pady=10)
        
        # Display first plot
        self.show_plot(0)
    
    def show_plot(self, index):
        if 0 <= index < len(self.plot_files):
            self.current_index = index
            file_path = os.path.join(self.plot_dir, self.plot_files[index])
            
            # Open and resize image
            img = Image.open(file_path)
            
            # Calculate new size while maintaining aspect ratio
            width, height = img.size
            max_width = 1100
            max_height = 650
            
            # Scale down if needed
            if width > max_width or height > max_height:
                ratio = min(max_width / width, max_height / height)
                width = int(width * ratio)
                height = int(height * ratio)
                img = img.resize((width, height), Image.LANCZOS)
            
            # Convert to Tkinter format
            tk_img = ImageTk.PhotoImage(img)
            self.img_label.configure(image=tk_img)
            self.img_label.image = tk_img  # Keep a reference
            
            # Update description
            plot_name = self.plot_files[index]
            description = self.plot_descriptions.get(plot_name, "No description available")
            self.desc_label.configure(text=f"{plot_name}: {description}")
            
            # Update counter
            self.counter_label.configure(text=f"Plot {index+1} of {len(self.plot_files)}")
            
            # Update dropdown
            self.plot_var.set(f"{index+1}. {plot_name}")
    
    def show_next(self):
        self.show_plot((self.current_index + 1) % len(self.plot_files))
    
    def show_previous(self):
        self.show_plot((self.current_index - 1) % len(self.plot_files))
    
    def on_plot_selected(self, event):
        selected = self.plot_var.get()
        index = int(selected.split('.')[0]) - 1
        self.show_plot(index)

if __name__ == "__main__":
    root = tk.Tk()
    app = PlotViewer(root, "plots")
    root.mainloop() 