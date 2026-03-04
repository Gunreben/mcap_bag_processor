#!/usr/bin/env python3
"""
MCAP Bag Processor - GUI Application

Modern tkinter GUI for offline MCAP bag batch processing.
"""

import glob
import os
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List

from .preferences import load_preferences, save_preferences
from .processor import (
    McapBagProcessor, ProcessorConfig, DEFAULT_TOPICS,
    generate_rosbag2_metadata,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BG = "#f0f0f0"
CARD_BG = "#ffffff"
ACCENT = "#1a73e8"
ACCENT_HOVER = "#1557b0"
TEXT = "#202124"
TEXT_SECONDARY = "#5f6368"
BORDER = "#dadce0"
LOG_BG = "#1e1e2e"
LOG_FG = "#cdd6f4"


def _configure_styles():
    """Apply a flat, modern look on top of the 'clam' theme."""
    style = ttk.Style()
    style.theme_use("clam")

    style.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 10))
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=CARD_BG, relief="solid", borderwidth=1)
    style.configure(
        "TLabelframe", background=CARD_BG, relief="solid", borderwidth=1,
    )
    style.configure(
        "TLabelframe.Label", background=CARD_BG, foreground=TEXT,
        font=("Segoe UI", 10, "bold"),
    )
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Card.TLabel", background=CARD_BG)
    style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"), background=BG)
    style.configure("Secondary.TLabel", foreground=TEXT_SECONDARY, background=BG)

    style.configure("TEntry", fieldbackground="white", borderwidth=1, relief="solid")

    style.configure(
        "Accent.TButton", font=("Segoe UI", 10, "bold"),
        background=ACCENT, foreground="white", borderwidth=0, padding=(16, 6),
    )
    style.map(
        "Accent.TButton",
        background=[("active", ACCENT_HOVER), ("disabled", BORDER)],
        foreground=[("disabled", TEXT_SECONDARY)],
    )
    style.configure("TButton", padding=(10, 4))

    style.configure("TCheckbutton", background=CARD_BG, foreground=TEXT)
    style.map("TCheckbutton", background=[("active", CARD_BG)])

    style.configure(
        "Horizontal.TProgressbar",
        troughcolor=BORDER, background=ACCENT, thickness=8,
    )


# ---------------------------------------------------------------------------
# Preferences dialog
# ---------------------------------------------------------------------------

class PreferencesDialog(tk.Toplevel):
    """Modal dialog for editing persistent default paths."""

    def __init__(self, parent: tk.Tk):
        super().__init__(parent)
        self.title("Preferences")
        self.geometry("620x200")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.prefs = load_preferences()
        self._urdf = tk.StringVar(value=self.prefs.get("urdf_path", ""))
        self._calib = tk.StringVar(value=self.prefs.get("calibration_dir", ""))

        body = ttk.Frame(self, padding=16)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(1, weight=1)

        ttk.Label(body, text="Default URDF File:").grid(
            row=0, column=0, sticky="w", pady=4,
        )
        ttk.Entry(body, textvariable=self._urdf).grid(
            row=0, column=1, sticky="ew", padx=(8, 4), pady=4,
        )
        ttk.Button(body, text="Browse...", command=self._browse_urdf).grid(
            row=0, column=2, pady=4,
        )

        ttk.Label(body, text="Default Calibration Dir:").grid(
            row=1, column=0, sticky="w", pady=4,
        )
        ttk.Entry(body, textvariable=self._calib).grid(
            row=1, column=1, sticky="ew", padx=(8, 4), pady=4,
        )
        ttk.Button(body, text="Browse...", command=self._browse_calib).grid(
            row=1, column=2, pady=4,
        )

        btn_row = ttk.Frame(body)
        btn_row.grid(row=2, column=0, columnspan=3, pady=(16, 0), sticky="e")
        ttk.Button(btn_row, text="Cancel", command=self.destroy).pack(
            side=tk.RIGHT, padx=(8, 0),
        )
        ttk.Button(
            btn_row, text="Save", style="Accent.TButton", command=self._save,
        ).pack(side=tk.RIGHT)

    # -- Browse helpers --

    def _browse_urdf(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Select Default URDF",
            filetypes=[("URDF files", "*.urdf"), ("All files", "*.*")],
        )
        if path:
            self._urdf.set(path)

    def _browse_calib(self):
        path = filedialog.askdirectory(parent=self, title="Select Default Calibration Directory")
        if path:
            self._calib.set(path)

    def _save(self):
        self.prefs["urdf_path"] = self._urdf.get()
        self.prefs["calibration_dir"] = self._calib.get()
        save_preferences(self.prefs)
        self.destroy()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class McapProcessorGUI:
    """Main GUI for batch MCAP bag processing."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MCAP Bag Processor")
        self.root.geometry("850x720")
        self.root.minsize(750, 640)
        self.root.configure(bg=BG)

        _configure_styles()

        # -- tk variables --
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.urdf_path = tk.StringVar()
        self.calibration_dir = tk.StringVar()

        self.generate_tf_static = tk.BooleanVar(value=True)
        self.add_map_odom_tf = tk.BooleanVar(value=True)
        self.generate_camera_info = tk.BooleanVar(value=True)
        self.filter_pointcloud = tk.BooleanVar(value=True)
        self.generate_lidar_odom = tk.BooleanVar(value=False)
        self.fix_ouster_timestamps = tk.BooleanVar(value=False)

        self.processing = False

        self._load_prefs()
        self._build_ui()

    # -- Preferences --

    def _load_prefs(self):
        prefs = load_preferences()
        if prefs.get("urdf_path"):
            self.urdf_path.set(prefs["urdf_path"])
        if prefs.get("calibration_dir"):
            self.calibration_dir.set(prefs["calibration_dir"])
        if prefs.get("last_input_dir"):
            self.input_dir.set(prefs["last_input_dir"])
        if prefs.get("last_output_dir"):
            self.output_dir.set(prefs["last_output_dir"])

    def _persist_dirs(self):
        """Save last-used input/output directories."""
        prefs = load_preferences()
        prefs["last_input_dir"] = self.input_dir.get()
        prefs["last_output_dir"] = self.output_dir.get()
        save_preferences(prefs)

    def _open_preferences(self):
        PreferencesDialog(self.root)
        self._load_prefs()

    # -- UI construction --

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)

        row = 0

        # ---- Title bar ----
        title_frame = ttk.Frame(outer)
        title_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        title_frame.columnconfigure(0, weight=1)

        ttk.Label(title_frame, text="MCAP Offline Bag Processor", style="Title.TLabel").grid(
            row=0, column=0, sticky="w",
        )
        ttk.Button(title_frame, text="Preferences...", command=self._open_preferences).grid(
            row=0, column=1, sticky="e",
        )

        row += 1

        # ---- Input / Output ----
        io = ttk.LabelFrame(outer, text="  Input / Output  ", padding=10)
        io.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        io.columnconfigure(1, weight=1)

        ttk.Label(io, text="Input Directory:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(io, textvariable=self.input_dir).grid(
            row=0, column=1, sticky="ew", padx=6, pady=3,
        )
        ttk.Button(io, text="Browse...", command=self._browse_input).grid(row=0, column=2, pady=3)

        ttk.Label(io, text="Output Directory:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(io, textvariable=self.output_dir).grid(
            row=1, column=1, sticky="ew", padx=6, pady=3,
        )
        ttk.Button(io, text="Browse...", command=self._browse_output).grid(row=1, column=2, pady=3)

        row += 1

        # ---- Configuration ----
        cfg = ttk.LabelFrame(outer, text="  Configuration  ", padding=10)
        cfg.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        cfg.columnconfigure(1, weight=1)

        ttk.Label(cfg, text="URDF File:").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(cfg, textvariable=self.urdf_path).grid(
            row=0, column=1, sticky="ew", padx=6, pady=3,
        )
        ttk.Button(cfg, text="Browse...", command=self._browse_urdf).grid(row=0, column=2, pady=3)

        ttk.Label(cfg, text="Calibration Dir:").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(cfg, textvariable=self.calibration_dir).grid(
            row=1, column=1, sticky="ew", padx=6, pady=3,
        )
        ttk.Button(cfg, text="Browse...", command=self._browse_calibration).grid(
            row=1, column=2, pady=3,
        )

        row += 1

        # ---- Processing options ----
        opts = ttk.LabelFrame(outer, text="  Processing Options  ", padding=10)
        opts.grid(row=row, column=0, sticky="ew", pady=(0, 6))

        opt_grid = ttk.Frame(opts)
        opt_grid.pack(fill=tk.X)

        ttk.Checkbutton(opt_grid, text="Generate tf_static from URDF", variable=self.generate_tf_static).grid(
            row=0, column=0, sticky="w", padx=(0, 24), pady=2,
        )
        ttk.Checkbutton(opt_grid, text="Add map->odom->base_link TF", variable=self.add_map_odom_tf).grid(
            row=0, column=1, sticky="w", pady=2,
        )
        ttk.Checkbutton(opt_grid, text="Generate camera_info for images", variable=self.generate_camera_info).grid(
            row=1, column=0, sticky="w", padx=(0, 24), pady=2,
        )
        ttk.Checkbutton(opt_grid, text="Filter ZED pointcloud (alpha + outlier)", variable=self.filter_pointcloud).grid(
            row=1, column=1, sticky="w", pady=2,
        )
        ttk.Checkbutton(opt_grid, text="Generate LiDAR odometry (KISS-ICP)", variable=self.generate_lidar_odom).grid(
            row=2, column=0, sticky="w", pady=2,
        )
        ttk.Checkbutton(opt_grid, text="Fix Ouster timestamps (internal osc → wall clock)", variable=self.fix_ouster_timestamps).grid(
            row=2, column=1, sticky="w", pady=2,
        )

        row += 1

        # ---- Progress ----
        prog = ttk.LabelFrame(outer, text="  Progress  ", padding=10)
        prog.grid(row=row, column=0, sticky="ew", pady=(0, 6))

        self.progress_bar = ttk.Progressbar(prog, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(0, 4))

        self.status_label = ttk.Label(prog, text="Ready", style="Secondary.TLabel")
        self.status_label.pack(anchor=tk.W)

        row += 1

        # ---- Log ----
        log_frame = ttk.LabelFrame(outer, text="  Log  ", padding=6)
        log_frame.grid(row=row, column=0, sticky="nsew", pady=(0, 8))
        outer.rowconfigure(row, weight=1)

        log_inner = ttk.Frame(log_frame)
        log_inner.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            log_inner, height=8, wrap=tk.WORD, state=tk.DISABLED,
            bg=LOG_BG, fg=LOG_FG, font=("Consolas", 9),
            insertbackground=LOG_FG, selectbackground=ACCENT,
            borderwidth=0, highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(log_inner, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        row += 1

        # ---- Buttons ----
        btn_frame = ttk.Frame(outer)
        btn_frame.grid(row=row, column=0, sticky="ew")

        self.process_button = ttk.Button(
            btn_frame, text="Process All Bags", style="Accent.TButton",
            command=self._start_processing,
        )
        self.process_button.pack(side=tk.RIGHT, padx=(6, 0))

        ttk.Button(btn_frame, text="Show Topics", command=self._show_topics).pack(
            side=tk.RIGHT,
        )

    # -- Browse helpers --

    def _browse_input(self):
        path = filedialog.askdirectory(title="Select Input Directory")
        if path:
            self.input_dir.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select Output Directory")
        if path:
            self.output_dir.set(path)

    def _browse_urdf(self):
        path = filedialog.askopenfilename(
            title="Select URDF File",
            filetypes=[("URDF files", "*.urdf"), ("XML files", "*.xml"), ("All files", "*.*")],
        )
        if path:
            self.urdf_path.set(path)

    def _browse_calibration(self):
        path = filedialog.askdirectory(title="Select Calibration Directory")
        if path:
            self.calibration_dir.set(path)

    # -- Logging / progress --

    def _log(self, message: str):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _update_progress(self, progress: float, message: str):
        self.progress_bar["value"] = progress * 100
        self.status_label.configure(text=message)
        self.root.update_idletasks()

    # -- Show topics --

    def _find_mcap_files(self) -> List[str]:
        input_dir = self.input_dir.get()
        if not input_dir or not os.path.isdir(input_dir):
            return []
        return sorted(glob.glob(os.path.join(input_dir, "*.mcap")))

    def _show_topics(self):
        files = self._find_mcap_files()
        if not files:
            messagebox.showerror("Error", "No .mcap files found in the input directory.")
            return
        first = files[0]
        try:
            config = ProcessorConfig(input_path=first, output_path="")
            processor = McapBagProcessor(config)
            topics = processor.get_input_topics()

            self._log(f"\n=== Topics in {os.path.basename(first)} ===")
            for topic in topics:
                marker = "+" if topic in DEFAULT_TOPICS else "-"
                self._log(f"  [{marker}] {topic}")
            self._log(f"Total: {len(topics)} topics")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read bag: {e}")

    # -- Validation --

    def _validate_inputs(self) -> bool:
        if not self.input_dir.get():
            messagebox.showerror("Error", "Please select an input directory.")
            return False
        if not os.path.isdir(self.input_dir.get()):
            messagebox.showerror("Error", "Input directory does not exist.")
            return False
        if not self._find_mcap_files():
            messagebox.showerror("Error", "No .mcap files found in the input directory.")
            return False
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please select an output directory.")
            return False
        if self.generate_tf_static.get():
            if not self.urdf_path.get():
                messagebox.showerror("Error", "Please select a URDF file for tf_static generation.")
                return False
            if not os.path.exists(self.urdf_path.get()):
                messagebox.showerror("Error", "URDF file does not exist.")
                return False
        if self.generate_camera_info.get():
            if not self.calibration_dir.get():
                messagebox.showerror("Error", "Please select a calibration directory.")
                return False
            if not os.path.isdir(self.calibration_dir.get()):
                messagebox.showerror("Error", "Calibration directory does not exist.")
                return False
        return True

    # -- Batch processing --

    def _start_processing(self):
        if self.processing:
            return
        if not self._validate_inputs():
            return

        self._persist_dirs()

        mcap_files = self._find_mcap_files()
        output_dir = self.output_dir.get()
        os.makedirs(output_dir, exist_ok=True)

        self.processing = True
        self.process_button.configure(state=tk.DISABLED)

        self._log(f"\n=== Batch processing {len(mcap_files)} bag(s) ===")

        thread = threading.Thread(
            target=self._batch_thread, args=(mcap_files, output_dir), daemon=True,
        )
        thread.start()

    def _batch_thread(self, mcap_files: List[str], output_dir: str):
        total_files = len(mcap_files)
        failed = 0
        all_stats = []
        # Persist LiDAR odometry across bags so pose + map carry over
        shared_lidar_odom = None

        for idx, mcap_path in enumerate(mcap_files):
            bag_name = os.path.basename(mcap_path)
            if bag_name.endswith(".mcap"):
                out_name = bag_name[:-5] + "_processed.mcap"
            else:
                out_name = bag_name + "_processed.mcap"
            output_path = os.path.join(output_dir, out_name)

            self.root.after(0, lambda b=bag_name, i=idx: self._log(
                f"\n--- [{i + 1}/{total_files}] {b} ---",
            ))

            config = ProcessorConfig(
                input_path=mcap_path,
                output_path=output_path,
                urdf_path=self.urdf_path.get() if self.generate_tf_static.get() else None,
                calibration_dir=self.calibration_dir.get() if self.generate_camera_info.get() else None,
                generate_tf_static=self.generate_tf_static.get(),
                generate_camera_info=self.generate_camera_info.get(),
                filter_pointcloud=self.filter_pointcloud.get(),
                add_map_odom_tf=self.add_map_odom_tf.get(),
                generate_lidar_odom=self.generate_lidar_odom.get(),
                fix_ouster_timestamps=self.fix_ouster_timestamps.get(),
            )

            file_idx = idx

            def _progress_cb(p, m, _fi=file_idx, _tf=total_files):
                overall = (_fi + p) / _tf
                msg = f"Bag {_fi + 1}/{_tf} - {m}"
                self.root.after(0, lambda o=overall, s=msg: self._update_progress(o, s))

            try:
                processor = McapBagProcessor(config, lidar_odom=shared_lidar_odom)
                processor.set_progress_callback(_progress_cb)
                stats = processor.process()
                all_stats.append(stats)
                if processor.lidar_odom is not None:
                    shared_lidar_odom = processor.lidar_odom

                def _log_stats(s=stats):
                    parts = [
                        f"Messages: {s.total_messages}",
                        f"Passthrough: {s.passthrough_messages}",
                        f"CameraInfo: {s.generated_camera_info}",
                        f"Filtered PC: {s.filtered_pointclouds}",
                        f"LiDAR odom: {s.generated_lidar_odom}",
                    ]
                    if s.patched_ouster_timestamps:
                        parts.append(f"Ouster ts fixed: {s.patched_ouster_timestamps}")
                    self._log("  " + "  |  ".join(parts))

                self.root.after(0, _log_stats)
            except Exception as exc:
                failed += 1
                tb = traceback.format_exc()
                self.root.after(0, lambda e=exc, t=tb: self._log(f"  ERROR: {e}\n{t}"))

        # Generate rosbag2 metadata.yaml so the output directory is a valid bag
        if all_stats:
            try:
                meta_path = generate_rosbag2_metadata(output_dir, all_stats)
                self.root.after(0, lambda p=meta_path: self._log(
                    f"  Wrote {p}"
                ))
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._log(
                    f"  Warning: could not write metadata.yaml: {e}"
                ))

        self.root.after(0, lambda: self._batch_complete(total_files, failed))

    def _batch_complete(self, total: int, failed: int):
        self.processing = False
        self.process_button.configure(state=tk.NORMAL)
        self._update_progress(1.0, "Done")

        ok = total - failed
        self._log(f"\n=== Batch complete: {ok}/{total} succeeded ===")
        if failed:
            messagebox.showwarning("Done", f"{ok}/{total} bags processed successfully.\n{failed} failed (see log).")
        else:
            messagebox.showinfo("Done", f"All {total} bags processed successfully.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    root = tk.Tk()
    _app = McapProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
