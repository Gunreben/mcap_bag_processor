#!/usr/bin/env python3
"""
MCAP Bag Processor - GUI Application

A simple tkinter GUI for offline MCAP bag processing that adds:
- tf_static from URDF
- camera_info matched to image timestamps  
- filtered ZED pointclouds
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional

from .processor import McapBagProcessor, ProcessorConfig, DEFAULT_TOPICS


class McapProcessorGUI:
    """Main GUI application for MCAP bag processing."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MCAP Bag Processor")
        self.root.geometry("800x700")
        self.root.minsize(700, 600)
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'))
        self.style.configure('Section.TLabelframe.Label', font=('Helvetica', 10, 'bold'))
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.urdf_path = tk.StringVar()
        self.calibration_dir = tk.StringVar()
        
        self.generate_tf_static = tk.BooleanVar(value=True)
        self.generate_camera_info = tk.BooleanVar(value=True)
        self.filter_pointcloud = tk.BooleanVar(value=True)
        self.add_map_odom_tf = tk.BooleanVar(value=True)
        
        # Processing state
        self.processing = False
        self.processor: Optional[McapBagProcessor] = None
        
        # Set default paths based on workspace
        self._set_default_paths()
        
        # Build UI
        self._create_widgets()
    
    def _set_default_paths(self):
        """Set default paths based on ROS workspace."""
        # Try to find the workspace root
        script_dir = Path(__file__).parent.parent.parent
        
        # Default URDF path
        urdf_path = script_dir / 'vario700_sensorrig' / 'urdf' / 'vario700_sensorrig_msa.urdf'
        if urdf_path.exists():
            self.urdf_path.set(str(urdf_path))
        
        # Default calibration directory
        calib_dir = script_dir / 'tractor_multi_cam_publisher' / 'calibration'
        if calib_dir.exists():
            self.calibration_dir.set(str(calib_dir))
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="MCAP Offline Bag Processor",
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 10))
        
        # === Input/Output Section ===
        io_frame = ttk.LabelFrame(main_frame, text="Input / Output", style='Section.TLabelframe', padding="10")
        io_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input bag
        input_row = ttk.Frame(io_frame)
        input_row.pack(fill=tk.X, pady=2)
        ttk.Label(input_row, text="Input MCAP:", width=15).pack(side=tk.LEFT)
        ttk.Entry(input_row, textvariable=self.input_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(input_row, text="Browse...", command=self._browse_input).pack(side=tk.LEFT)
        
        # Output directory
        output_row = ttk.Frame(io_frame)
        output_row.pack(fill=tk.X, pady=2)
        ttk.Label(output_row, text="Output Directory:", width=15).pack(side=tk.LEFT)
        ttk.Entry(output_row, textvariable=self.output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(output_row, text="Browse...", command=self._browse_output).pack(side=tk.LEFT)
        
        # === Configuration Section ===
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", style='Section.TLabelframe', padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # URDF path
        urdf_row = ttk.Frame(config_frame)
        urdf_row.pack(fill=tk.X, pady=2)
        ttk.Label(urdf_row, text="URDF File:", width=15).pack(side=tk.LEFT)
        ttk.Entry(urdf_row, textvariable=self.urdf_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(urdf_row, text="Browse...", command=self._browse_urdf).pack(side=tk.LEFT)
        
        # Calibration directory
        calib_row = ttk.Frame(config_frame)
        calib_row.pack(fill=tk.X, pady=2)
        ttk.Label(calib_row, text="Calibration Dir:", width=15).pack(side=tk.LEFT)
        ttk.Entry(calib_row, textvariable=self.calibration_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(calib_row, text="Browse...", command=self._browse_calibration).pack(side=tk.LEFT)
        
        # === Options Section ===
        options_frame = ttk.LabelFrame(main_frame, text="Processing Options", style='Section.TLabelframe', padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        options_row1 = ttk.Frame(options_frame)
        options_row1.pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(
            options_row1, text="Generate tf_static from URDF",
            variable=self.generate_tf_static
        ).pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Checkbutton(
            options_row1, text="Add map->odom->base_link TF",
            variable=self.add_map_odom_tf
        ).pack(side=tk.LEFT)
        
        options_row2 = ttk.Frame(options_frame)
        options_row2.pack(fill=tk.X, pady=2)
        
        ttk.Checkbutton(
            options_row2, text="Generate camera_info for images",
            variable=self.generate_camera_info
        ).pack(side=tk.LEFT, padx=(0, 20))
        
        ttk.Checkbutton(
            options_row2, text="Filter ZED pointcloud (alpha + outlier)",
            variable=self.filter_pointcloud
        ).pack(side=tk.LEFT)
        
        # === Progress Section ===
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", style='Section.TLabelframe', padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(anchor=tk.W)
        
        # === Log Section ===
        log_frame = ttk.LabelFrame(main_frame, text="Log", style='Section.TLabelframe', padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Text widget with scrollbar
        log_container = ttk.Frame(log_frame)
        log_container.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_container, height=10, wrap=tk.WORD, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(log_container, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # === Buttons Section ===
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        self.process_button = ttk.Button(
            button_frame, text="Process Bag",
            command=self._start_processing
        )
        self.process_button.pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(
            button_frame, text="Show Input Topics",
            command=self._show_topics
        ).pack(side=tk.RIGHT, padx=5)
    
    def _browse_input(self):
        """Browse for input MCAP file."""
        # Look for .mcap file or directory containing mcap
        path = filedialog.askopenfilename(
            title="Select Input MCAP",
            filetypes=[
                ("MCAP files", "*.mcap"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.input_path.set(path)
            # Auto-set output directory
            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(path))
    
    def _browse_output(self):
        """Browse for output directory."""
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir.set(path)
    
    def _browse_urdf(self):
        """Browse for URDF file."""
        path = filedialog.askopenfilename(
            title="Select URDF File",
            filetypes=[
                ("URDF files", "*.urdf"),
                ("XML files", "*.xml"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.urdf_path.set(path)
    
    def _browse_calibration(self):
        """Browse for calibration directory."""
        path = filedialog.askdirectory(title="Select Calibration Directory")
        if path:
            self.calibration_dir.set(path)
    
    def _log(self, message: str):
        """Add message to log."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
    
    def _update_progress(self, progress: float, message: str):
        """Update progress bar and status."""
        self.progress_bar['value'] = progress * 100
        self.status_label.configure(text=message)
        self.root.update_idletasks()
    
    def _show_topics(self):
        """Show topics in input bag."""
        input_path = self.input_path.get()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid input MCAP file")
            return
        
        try:
            config = ProcessorConfig(
                input_path=input_path,
                output_path=""
            )
            processor = McapBagProcessor(config)
            topics = processor.get_input_topics()
            
            self._log(f"\n=== Topics in {os.path.basename(input_path)} ===")
            for topic in topics:
                included = "+" if topic in DEFAULT_TOPICS else "-"
                self._log(f"  [{included}] {topic}")
            self._log(f"Total: {len(topics)} topics")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read bag: {e}")
    
    def _validate_inputs(self) -> bool:
        """Validate all inputs before processing."""
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input MCAP file")
            return False
        
        if not os.path.exists(self.input_path.get()):
            messagebox.showerror("Error", "Input MCAP file does not exist")
            return False
        
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory")
            return False
        
        if self.generate_tf_static.get() and not self.urdf_path.get():
            messagebox.showerror("Error", "Please select a URDF file for tf_static generation")
            return False
        
        if self.generate_tf_static.get() and not os.path.exists(self.urdf_path.get()):
            messagebox.showerror("Error", "URDF file does not exist")
            return False
        
        if self.generate_camera_info.get() and not self.calibration_dir.get():
            messagebox.showerror("Error", "Please select a calibration directory")
            return False
        
        if self.generate_camera_info.get() and not os.path.isdir(self.calibration_dir.get()):
            messagebox.showerror("Error", "Calibration directory does not exist")
            return False
        
        return True
    
    def _start_processing(self):
        """Start bag processing in background thread."""
        if self.processing:
            return
        
        if not self._validate_inputs():
            return
        
        # Generate output filename
        input_basename = os.path.basename(self.input_path.get())
        if input_basename.endswith('.mcap'):
            output_name = input_basename[:-5] + '_processed.mcap'
        else:
            output_name = input_basename + '_processed.mcap'
        
        output_path = os.path.join(self.output_dir.get(), output_name)
        
        # Confirm overwrite if exists
        if os.path.exists(output_path):
            if not messagebox.askyesno("Confirm", f"Output file already exists:\n{output_path}\n\nOverwrite?"):
                return
        
        # Create config
        config = ProcessorConfig(
            input_path=self.input_path.get(),
            output_path=output_path,
            urdf_path=self.urdf_path.get() if self.generate_tf_static.get() else None,
            calibration_dir=self.calibration_dir.get() if self.generate_camera_info.get() else None,
            generate_tf_static=self.generate_tf_static.get(),
            generate_camera_info=self.generate_camera_info.get(),
            filter_pointcloud=self.filter_pointcloud.get(),
            add_map_odom_tf=self.add_map_odom_tf.get()
        )
        
        self.processing = True
        self.process_button.configure(state=tk.DISABLED)
        
        self._log(f"\n=== Starting processing ===")
        self._log(f"Input: {config.input_path}")
        self._log(f"Output: {config.output_path}")
        
        # Run in background thread
        thread = threading.Thread(target=self._process_thread, args=(config,))
        thread.daemon = True
        thread.start()
    
    def _process_thread(self, config: ProcessorConfig):
        """Background thread for processing."""
        try:
            processor = McapBagProcessor(config)
            processor.set_progress_callback(
                lambda p, m: self.root.after(0, lambda p=p, m=m: self._update_progress(p, m))
            )
            
            stats = processor.process()
            
            # Update UI on main thread
            self.root.after(0, lambda s=stats: self._processing_complete(s))
            
        except Exception as e:
            import traceback
            error_msg = f"{e}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda err=error_msg: self._processing_error(err))
    
    def _processing_complete(self, stats):
        """Handle processing completion."""
        self.processing = False
        self.process_button.configure(state=tk.NORMAL)
        
        self._log("\n=== Processing Complete ===")
        self._log(f"Total messages: {stats.total_messages}")
        self._log(f"Passthrough: {stats.passthrough_messages}")
        self._log(f"Camera info generated: {stats.generated_camera_info}")
        self._log(f"Pointclouds filtered: {stats.filtered_pointclouds}")
        self._log(f"Skipped (tf): {stats.skipped_messages}")
        
        messagebox.showinfo("Success", "Bag processing completed successfully!")
    
    def _processing_error(self, error: str):
        """Handle processing error."""
        self.processing = False
        self.process_button.configure(state=tk.NORMAL)
        
        self._log(f"\n=== ERROR ===")
        self._log(error)
        
        messagebox.showerror("Error", f"Processing failed:\n{error}")


def main():
    """Main entry point."""
    root = tk.Tk()
    app = McapProcessorGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

