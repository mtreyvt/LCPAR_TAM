#!/usr/bin/env python3
"""
Graphical user interface for the Low‑Cost Portable Antenna Range radio system.

This module provides a Tkinter‐based GUI that wraps the existing command
line functionality in ``main.py`` and ``RadioFunctions.py``.  Users can
initiate antenna measurements and scans, load parameter files, capture
single samples, and visualize results without interacting through the
terminal.  Results and status messages are streamed into a log box so
long running operations do not block the interface.

The GUI mirrors the menu options from the original CLI:

1. FastScan (coherent AM, rotating)
2. Measure (coherent AM, stepwise)
3. Measure (Noise Subtraction)
4. Plot last run data
5. Plot data from file
6. Plot data from two files
7. Capture single measurement (Tx ON)
8. Capture single background (Tx OFF)
9. Time‑Gated Measure (coherent AM, stepwise)
10. Time‑Gated FastScan (rotating, freq sweep)
11. Time‑Gated single pointing (freq sweep, no rotation)
12. Time‑Gated single frequency @ 2.5 GHz (rotating)
13. Quit

Selecting an operation spawns a worker thread so the GUI remains
responsive.  Standard output from the underlying functions is captured
and appended to the log panel.  When appropriate a measurement result
is stored internally for later plotting.
"""

from __future__ import annotations

import os
import io
import threading
import queue
import contextlib
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from tkinter.scrolledtext import ScrolledText

# Import project modules.  These imports assume this script is placed
# alongside ``main.py`` in the ``radio_tg`` package.  If you move this
# file elsewhere adjust the import paths accordingly.
try:
    import RadioFunctions
    from PlotGraph import PlotGraph
    from PolarPlot import plot_polar_patterns
except ImportError as exc:
    raise ImportError(
        "Failed to import project modules. Ensure radio_gui.py resides in the "
        "same package as main.py (e.g. radio_tg) and that the parent directory "
        "is on PYTHONPATH."
    ) from exc

class RadioGUI:
    """Main application class for the radio system GUI."""

    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.master.title("LCPAR Radio System GUI")

        # Path to the JSON parameter file.  Default uses params.json in the
        # working directory.  Users may browse to a different file.
        self.param_file: str = os.path.join(os.getcwd(), "params.json")
        self.last_data = None  # stores antenna data from the last scan/measure

        # Create a queue for thread‑safe communication between worker threads
        # and the GUI.  Each message in the queue is a string to be appended
        # to the log panel.
        self._log_queue: queue.Queue[str] = queue.Queue()

        # Build the user interface
        self._build_ui()

        # Start periodic polling of the queue to update the log panel
        self.master.after(100, self._process_queue)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        """Construct all widgets for the GUI."""
        # Frame for parameter file selection
        param_frame = ttk.LabelFrame(self.master, text="Parameters")
        param_frame.pack(fill=tk.X, padx=10, pady=5)

        param_label = ttk.Label(param_frame, text="Parameter file:")
        param_label.pack(side=tk.LEFT, padx=(10, 5), pady=5)

        self.param_var = tk.StringVar(value=self.param_file)
        param_entry = ttk.Entry(param_frame, textvariable=self.param_var, width=50)
        param_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        browse_btn = ttk.Button(param_frame, text="Browse…", command=self._browse_params)
        browse_btn.pack(side=tk.RIGHT, padx=(5, 10), pady=5)

        # Frame for the operation buttons
        btn_frame = ttk.LabelFrame(self.master, text="Operations")
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        # Define menu labels and associated handler methods
        menu_items = [
            ("FastScan (coherent AM, rotating)", self._handle_fastscan),
            ("Measure (coherent AM, stepwise)", self._handle_measure),
            ("Measure (Noise Subtraction)", self._handle_ns_measure),
            ("Plot last run data", self._handle_plot_last),
            ("Plot data from file", self._handle_plot_file),
            ("Plot data from two files", self._handle_plot_two_files),
            ("Capture single measurement (Tx ON)", self._handle_single_tx),
            ("Capture single background (Tx OFF)", self._handle_single_rx),
            ("Time‑Gated Measure (coherent AM, stepwise)", self._handle_am_tg_meas),
            ("Time‑Gated FastScan (rotating, freq sweep)", self._handle_am_tg_scan),
            ("Time‑Gated single pointing (freq sweep, no rotation)", self._handle_single_tg),
            ("Time‑Gated single frequency @ 2.5 GHz (rotating)", self._handle_single_freq_tg),
            ("Quit", self._quit_application),
        ]

        # Create buttons in a grid.  Layout 3 columns to fit nicely.
        cols = 3
        for idx, (label, handler) in enumerate(menu_items):
            row, col = divmod(idx, cols)
            btn = ttk.Button(btn_frame, text=label, command=handler)
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        # Make columns expand evenly
        for c in range(cols):
            btn_frame.columnconfigure(c, weight=1)

        # Log panel to display status and measurement outputs
        log_frame = ttk.LabelFrame(self.master, text="Output Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        self.log_text = ScrolledText(log_frame, wrap=tk.WORD, height=10, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # ----------------------------------------------------------------- Helpers
    def _browse_params(self) -> None:
        """Prompt the user to select a JSON parameter file."""
        filename = filedialog.askopenfilename(
            title="Select parameter JSON file",
            initialdir=os.path.dirname(self.param_file),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if filename:
            self.param_file = filename
            self.param_var.set(filename)
            self._append_log(f"Parameter file set to: {filename}")

    def _append_log(self, message: str) -> None:
        """Append a message to the log panel in a thread‑safe manner."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _process_queue(self) -> None:
        """Process all pending messages from worker threads."""
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass
        # Reschedule this method to run again after 100 ms
        self.master.after(100, self._process_queue)

    def _run_task(self, func, *, args: tuple = (), kwargs: dict | None = None,
                  description: str | None = None, callback=None) -> None:
        """
        Execute a RadioFunctions method in a background thread.

        * ``func`` is the callable to invoke.
        * ``args`` and ``kwargs`` are passed directly to ``func``.
        * ``description`` is a human readable name used in log entries.
        * ``callback`` is an optional function to call on the main thread
          with the return value from ``func``.
        """
        if kwargs is None:
            kwargs = {}
        desc = description or getattr(func, "__name__", "task")

        def worker():
            # Capture stdout from the function so we can display it in the log
            buffer = io.StringIO()
            self._log_queue.put(f"Starting {desc}…")
            try:
                with contextlib.redirect_stdout(buffer):
                    result = func(*args, **kwargs)
            except Exception as e:
                # Capture any printed output before the exception
                out = buffer.getvalue()
                if out:
                    for line in out.strip().splitlines():
                        self._log_queue.put(line)
                self._log_queue.put(f"Error during {desc}: {e}")
                return
            # Flush captured output
            out = buffer.getvalue()
            if out:
                for line in out.strip().splitlines():
                    self._log_queue.put(line)
            # Record return value and call callback on main thread
            if callback is not None:
                self.master.after(0, callback, result)
            # For simple scalar results, display them directly
            if result is not None and not isinstance(result, (list, tuple, dict)):
                self._log_queue.put(f"Result: {result}")
            self._log_queue.put(f"Finished {desc}.")

        threading.Thread(target=worker, daemon=True).start()

    # --------------------------------------------------------------- Handlers
    def _load_params(self) -> dict:
        """Load parameters from the selected file, falling back to defaults."""
        try:
            return RadioFunctions.LoadParams(self.param_file)
        except Exception as e:
            messagebox.showerror("Parameter Error", str(e))
            raise

    def _update_last_data(self, data) -> None:
        """Store measurement data for later plotting."""
        if data is not None:
            self.last_data = data
            self._append_log("Data captured and stored for plotting.")

    def _handle_fastscan(self) -> None:
        params = self._load_params()
        self._run_task(RadioFunctions.do_AMscan, args=(params,),
                       description="FastScan (AM scan)",
                       callback=self._update_last_data)

    def _handle_measure(self) -> None:
        params = self._load_params()
        self._run_task(RadioFunctions.do_AMmeas, args=(params,),
                       description="Measure (AM)",
                       callback=self._update_last_data)

    def _handle_ns_measure(self) -> None:
        params = self._load_params()
        if not hasattr(RadioFunctions, 'do_NSmeas'):
            messagebox.showerror("Not Implemented", "Noise subtraction measurement is not available.")
            return
        self._run_task(RadioFunctions.do_NSmeas, args=(params,),
                       description="Measure (Noise Subtraction)",
                       callback=self._update_last_data)

    def _handle_plot_last(self) -> None:
        if not self.last_data:
            messagebox.showinfo("No Data", "Run a scan or measurement before plotting.")
            return
        # Prompt for a title
        title = simpledialog.askstring("Plot Title", "Enter a title for the graph:")
        if not title:
            return
        def plot_task():
            pg = PlotGraph(self.last_data, title)
            pg.show()
        self._run_task(plot_task, description="Plotting last run data")

    def _handle_plot_file(self) -> None:
        filename = filedialog.askopenfilename(
            title="Select data file to plot",
            filetypes=[("CSV/Text files", "*.*")],
        )
        if not filename:
            return
        def plot_file_task():
            with open(filename, 'r') as fr:
                lines = fr.readlines()
            # skip notes and header
            if len(lines) >= 2:
                lines = lines[2:]
            file_data = RadioFunctions.get_plot_data(lines)
            pg = PlotGraph(file_data, os.path.basename(filename))
            pg.show()
        self._run_task(plot_file_task, description=f"Plotting {os.path.basename(filename)}")

    def _handle_plot_two_files(self) -> None:
        filenames = filedialog.askopenfilenames(
            title="Select two data files to plot",
            filetypes=[("CSV/Text files", "*.*")],
        )
        if not filenames or len(filenames) < 2:
            messagebox.showinfo("Selection Error", "Please select two files.")
            return
        file1, file2 = filenames[:2]
        def plot_two_task():
            # Load first file
            with open(file1, 'r') as fr:
                lines1 = fr.readlines()
            lines1 = lines1[2:] if len(lines1) >= 2 else lines1
            data1 = RadioFunctions.get_plot_data(lines1)
            pg1 = PlotGraph(data1, os.path.basename(file1))
            # Load second file
            with open(file2, 'r') as fr:
                lines2 = fr.readlines()
            lines2 = lines2[2:] if len(lines2) >= 2 else lines2
            data2 = RadioFunctions.get_plot_data(lines2)
            pg2 = PlotGraph(data2, os.path.basename(file2))
            # Generate a polar plot with both traces
            import numpy as np
            import matplotlib.pyplot as plt
            theta1 = np.deg2rad(pg1.mast_angles)
            theta2 = np.deg2rad(pg2.mast_angles)
            plt.figure(figsize=(7, 7))
            ax = plt.subplot(111, projection='polar')
            ax.set_theta_zero_location("N")
            ax.plot(theta1, pg1.rssi, label=os.path.basename(file1), linewidth=1.2)
            ax.plot(theta2, pg2.rssi, label=os.path.basename(file2), linewidth=1.2)
            ax.set_rticks([-20, -16, -12, -8, -4, 0])
            ax.legend(loc="best")
            ax.set_title("Comparison of two measurements", va='bottom')
            plt.show()
        self._run_task(plot_two_task, description="Plotting two files")

    def _handle_single_tx(self) -> None:
        # Capture a single measurement with transmitter on
        self._run_task(RadioFunctions.do_single, kwargs={"Tx": True},
                       description="Single measurement (Tx ON)")

    def _handle_single_rx(self) -> None:
        # Capture a single background measurement with transmitter off
        self._run_task(RadioFunctions.do_single, kwargs={"Tx": False},
                       description="Single background (Tx OFF)")

    def _handle_am_tg_meas(self) -> None:
        params = self._load_params()
        if not hasattr(RadioFunctions, 'do_AMTGmeas'):
            messagebox.showerror("Not Implemented", "Time‑Gated Measure function is not available.")
            return
        self._run_task(RadioFunctions.do_AMTGmeas, args=(params,),
                       description="Time‑Gated Measure", callback=self._update_last_data)

    def _handle_am_tg_scan(self) -> None:
        params = self._load_params()
        if not hasattr(RadioFunctions, 'do_AMTGscan'):
            messagebox.showerror("Not Implemented", "Time‑Gated FastScan function is not available.")
            return
        self._run_task(RadioFunctions.do_AMTGscan, args=(params,),
                       description="Time‑Gated FastScan", callback=self._update_last_data)

    def _handle_single_tg(self) -> None:
        params = self._load_params()
        if not hasattr(RadioFunctions, 'do_singleTG'):
            messagebox.showerror("Not Implemented", "Single pointing time‑gated function is not available.")
            return
        self._run_task(RadioFunctions.do_singleTG, args=(params,),
                       description="Single pointing TG", callback=self._update_last_data)

    def _handle_single_freq_tg(self) -> None:
        params = self._load_params()
        if not hasattr(RadioFunctions, 'do_AMTGscan_single_freq'):
            messagebox.showerror("Not Implemented", "Single frequency time‑gated scan function is not available.")
            return
        # default frequency 2.5 GHz and show_plots=True to mimic CLI
        self._run_task(RadioFunctions.do_AMTGscan_single_freq,
                       args=(params,),
                       kwargs={"freq_hz": 2.5e9, "show_plots": True},
                       description="Single frequency TG scan", callback=self._update_last_data)

    def _quit_application(self) -> None:
        """Close the application."""
        self.master.destroy()


def main() -> None:
    """Entry point when running this module as a script."""
    root = tk.Tk()
    app = RadioGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()