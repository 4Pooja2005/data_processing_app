import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from core.state import DatasetManager
from core.loader import load_file
from core.processor import remove_duplicates, handle_missing_values, standardize_data, merge_datasets
from core.utils import generate_temp_name

class DataProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Processing App")
        self.manager = DatasetManager()
        self.create_widgets()

    def create_widgets(self):
        # Upload button
        tk.Button(self.root, text="Upload Files", command=self.upload_files).pack(pady=5)

        # Dataset listbox (multi-select)
        self.listbox = tk.Listbox(self.root, height=6, selectmode=tk.MULTIPLE)
        self.listbox.pack(padx=5, pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)

        # Per-file operation buttons
        tk.Button(self.root, text="Remove Duplicates", command=self.remove_duplicates).pack(pady=2)
        tk.Button(self.root, text="Handle Missing Values", command=self.handle_missing_values).pack(pady=2)
        tk.Button(self.root, text="Standardize Data", command=self.standardize_data).pack(pady=2)

        # Cross-file operation
        tk.Button(self.root, text="Cross-file Merge", command=self.cross_file_merge).pack(pady=5)

        # Preview & Export
        tk.Button(self.root, text="Preview Data", command=self.preview_data).pack(pady=2)
        tk.Button(self.root, text="Export Dataset", command=self.export_dataset).pack(pady=2)

    # ---------------- Upload ----------------
    def upload_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV & Excel", "*.csv *.xlsx")])
        for path in paths:
            try:
                df = load_file(path)
                name = path.split("/")[-1]
                self.manager.add_dataset(name, df)
            except Exception as e:
                messagebox.showerror("Error", str(e))
        self.refresh_listbox()

    # ---------------- Listbox ----------------
    def refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for name in self.manager.datasets:
            self.listbox.insert(tk.END, name)

    def on_select(self, event):
        selection = self.listbox.curselection()
        if selection:
            # Set active dataset to first selected for per-file ops
            self.manager.active_dataset_name = self.listbox.get(selection[0])

    # ---------------- Per-file ops ----------------
    def remove_duplicates(self):
        self.manager.apply_basic_op(remove_duplicates)
        messagebox.showinfo("Done", "Duplicates removed.")

    def handle_missing_values(self):
        ds = self.manager.get_active_dataset()
        if not ds:
            return
        method = simpledialog.askstring("Missing Values", "Method: delete / zero / fill")
        fill_val = None
        if method == "fill":
            fill_val = simpledialog.askstring("Fill Value", "Enter value to fill missing:")
        ds.df = handle_missing_values(ds.df, method=method, fill_value=fill_val)
        messagebox.showinfo("Done", "Missing values handled.")

    def standardize_data(self):
        self.manager.apply_basic_op(standardize_data)
        messagebox.showinfo("Done", "Data standardized.")

    # ---------------- Cross-file ops ----------------
    def cross_file_merge(self):
        selected_indices = self.listbox.curselection()
        if not selected_indices or len(selected_indices) < 2:
            messagebox.showwarning("Selection Error", "Select at least two datasets")
            return
        names = [self.listbox.get(i) for i in selected_indices]
        temp_name = generate_temp_name("merged")
        self.manager.apply_cross_file_op(names, merge_datasets, temp_name)
        messagebox.showinfo("Done", f"Temporary dataset created: {temp_name}")
        self.refresh_listbox()

    # ---------------- Preview ----------------
    def preview_data(self):
        ds = self.manager.get_active_dataset()
        if not ds:
            messagebox.showwarning("No Dataset", "No active dataset")
            return
        top = tk.Toplevel(self.root)
        top.title(f"Preview: {ds.name}")
        text = tk.Text(top, width=100, height=30)
        text.pack()
        text.insert(tk.END, str(ds.df.head(20)))

    # ---------------- Export ----------------
    def export_dataset(self):
        ds = self.manager.get_active_dataset()
        if not ds:
            messagebox.showwarning("No Dataset", "No active dataset")
            return

        # Ask user where to save
        filetypes = [("CSV", "*.csv"), ("Excel", "*.xlsx"), ("JSON", "*.json")]
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=filetypes)

        if not save_path:
            return

        try:
            if save_path.endswith(".csv"):
                ds.df.to_csv(save_path, index=False)
            elif save_path.endswith(".xlsx"):
                ds.df.to_excel(save_path, index=False)
            elif save_path.endswith(".json"):
                ds.df.to_json(save_path, orient="records")
            else:
                messagebox.showerror("Error", "Unsupported file format")
                return
            messagebox.showinfo("Exported", f"Dataset saved as {save_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))
