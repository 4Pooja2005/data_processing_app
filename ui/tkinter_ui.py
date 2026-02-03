import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, simpledialog, Menu
from core.state import DatasetManager
from core.loader import load_file
from core.processor import remove_duplicates, handle_missing_values, standardize_data, standardize_column, merge_datasets
from core.utils import generate_temp_name

class DataProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Data Processing App")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        self.manager = DatasetManager()
        self.selection_order = []  # Track order of dataset selection
        self.create_widgets()

    def create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Data Processing Toolkit", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))

        # Upload section
        upload_frame = ttk.LabelFrame(main_frame, text="File Management", padding="10")
        upload_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(upload_frame, text="📁 Upload Files", command=self.upload_files).pack(side=tk.LEFT, padx=(0, 10))

        # Dataset selection
        select_frame = ttk.LabelFrame(main_frame, text="Dataset Selection", padding="10")
        select_frame.pack(fill=tk.X, pady=(0, 10))
        self.listbox = tk.Listbox(select_frame, height=4, selectmode=tk.MULTIPLE)
        self.listbox.pack(fill=tk.X)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.listbox.bind("<Button-3>", self.show_context_menu)

        # Operations sections
        ops_frame = ttk.Frame(main_frame)
        ops_frame.pack(fill=tk.X, pady=(0, 10))

        # Data Cleaning
        clean_frame = ttk.LabelFrame(ops_frame, text="Data Cleaning", padding="10")
        clean_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.remove_dup_btn = ttk.Button(clean_frame, text="🗑️ Remove Duplicates", command=self.remove_duplicates, state=tk.DISABLED)
        self.remove_dup_btn.pack(fill=tk.X, pady=2)
        self.handle_missing_btn = ttk.Button(clean_frame, text="🔧 Handle Missing Values", command=self.handle_missing_values, state=tk.DISABLED)
        self.handle_missing_btn.pack(fill=tk.X, pady=2)
        self.standardize_btn = ttk.Button(clean_frame, text="✨ Standardize Data", command=self.standardize_data, state=tk.DISABLED)
        self.standardize_btn.pack(fill=tk.X, pady=2)

        # History Management
        history_frame = ttk.LabelFrame(ops_frame, text="History", padding="10")
        history_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.undo_btn = ttk.Button(history_frame, text="↶ Undo", command=self.undo, state=tk.DISABLED)
        self.undo_btn.pack(fill=tk.X, pady=2)
        self.reset_btn = ttk.Button(history_frame, text="🔄 Reset", command=self.reset, state=tk.DISABLED)
        self.reset_btn.pack(fill=tk.X, pady=2)

        # Cross-file Operations
        cross_frame = ttk.LabelFrame(ops_frame, text="Cross-file Operations", padding="10")
        cross_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.merge_btn = ttk.Button(cross_frame, text="🔗 Cross-file Merge", command=self.cross_file_merge, state=tk.DISABLED)
        self.merge_btn.pack(fill=tk.X, pady=2)

        # View & Export
        view_frame = ttk.LabelFrame(ops_frame, text="View & Export", padding="10")
        view_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.preview_btn = ttk.Button(view_frame, text="👁️ Preview Data", command=self.preview_data, state=tk.DISABLED)
        self.preview_btn.pack(fill=tk.X, pady=2)
        self.export_btn = ttk.Button(view_frame, text="💾 Export Dataset", command=self.export_dataset, state=tk.DISABLED)
        self.export_btn.pack(fill=tk.X, pady=2)

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
        self.selection_order = []  # Reset selection order after refresh

    def on_select(self, event):
        current_selection = set(self.listbox.curselection())
        previous_selection = set(self.selection_order)
        
        newly_selected = current_selection - previous_selection
        newly_deselected = previous_selection - current_selection
        
        # Remove deselected
        for name in newly_deselected:
            if name in self.selection_order:
                self.selection_order.remove(name)
        
        # Add newly selected in order
        for idx in newly_selected:
            name = self.listbox.get(idx)
            if name not in self.selection_order:
                self.selection_order.append(name)
        
        selection = self.listbox.curselection()
        if selection:
            # Set active dataset to first selected for per-file ops
            self.manager.active_dataset_name = self.selection_order[0] if self.selection_order else None
            # Enable buttons
            self.remove_dup_btn.config(state=tk.NORMAL)
            self.handle_missing_btn.config(state=tk.NORMAL)
            self.standardize_btn.config(state=tk.NORMAL)
            self.preview_btn.config(state=tk.NORMAL)
            self.export_btn.config(state=tk.NORMAL)
            # Enable undo/reset if applicable
            ds = self.manager.get_active_dataset()
            if ds:
                self.reset_btn.config(state=tk.NORMAL)
                self.undo_btn.config(state=tk.NORMAL if len(ds.history) > 1 else tk.DISABLED)
            # Enable merge if at least 2 selected
            if len(self.selection_order) >= 2:
                self.merge_btn.config(state=tk.NORMAL)
            else:
                self.merge_btn.config(state=tk.DISABLED)
        else:
            # No selection, disable all
            self.manager.active_dataset_name = None
            self.selection_order = []
            self.remove_dup_btn.config(state=tk.DISABLED)
            self.handle_missing_btn.config(state=tk.DISABLED)
            self.standardize_btn.config(state=tk.DISABLED)
            self.merge_btn.config(state=tk.DISABLED)
            self.preview_btn.config(state=tk.DISABLED)
            self.export_btn.config(state=tk.DISABLED)
            self.undo_btn.config(state=tk.DISABLED)
            self.reset_btn.config(state=tk.DISABLED)

    def show_context_menu(self, event):
        # Get the index under the mouse
        index = self.listbox.nearest(event.y)
        if index < 0 or index >= self.listbox.size():
            return
        
        name = self.listbox.get(index)
        if name not in self.manager.datasets:
            return
        
        # Create context menu
        menu = Menu(self.root, tearoff=0)
        menu.add_command(label="Delete Dataset", command=lambda: self.delete_dataset(name))
        menu.post(event.x_root, event.y_root)

    def delete_dataset(self, name):
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{name}'?"):
            del self.manager.datasets[name]
            if name in self.selection_order:
                self.selection_order.remove(name)
            if self.manager.active_dataset_name == name:
                self.manager.active_dataset_name = self.selection_order[0] if self.selection_order else None
            self.refresh_listbox()
            self.on_select(None)  # Update button states
        ds = self.manager.get_active_dataset()
        if ds:
            self.undo_btn.config(state=tk.NORMAL if len(ds.history) > 1 else tk.DISABLED)
            self.reset_btn.config(state=tk.NORMAL)
        else:
            self.undo_btn.config(state=tk.DISABLED)
            self.reset_btn.config(state=tk.DISABLED)

    def undo(self):
        ds = self.manager.get_active_dataset()
        if ds and ds.undo():
            messagebox.showinfo("Done", "Last operation undone.")
            self.update_undo_buttons()
        else:
            messagebox.showwarning("Cannot Undo", "No operations to undo.")

    def reset(self):
        ds = self.manager.get_active_dataset()
        if ds and ds.reset():
            messagebox.showinfo("Done", "Dataset reset to original state.")
            self.update_undo_buttons()
        else:
            messagebox.showinfo("Already Reset", "Dataset is already in original state.")

    def remove_duplicates(self):
        ds = self.manager.get_active_dataset()
        if ds:
            new_df = remove_duplicates(ds.df)
            new_name = generate_temp_name(base="deduped")
            self.manager.add_dataset(new_name, new_df, temporary=True)
            self.refresh_listbox()
            # Optionally, select the new dataset
            idx = list(self.manager.datasets.keys()).index(new_name)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self.listbox.activate(idx)
            self.manager.active_dataset_name = new_name
            messagebox.showinfo("Done", f"Duplicates removed. New dataset '{new_name}' created.")
            self.update_undo_buttons()

    def handle_missing_values(self):
        ds = self.manager.get_active_dataset()
        if not ds:
            return
        
        # Create a custom dialog for method selection
        dialog = tk.Toplevel(self.root)
        dialog.title("Handle Missing Values")
        dialog.geometry("300x150")
        dialog.resizable(False, False)
        
        tk.Label(dialog, text="Choose method to handle missing values:").pack(pady=10)
        
        method_var = tk.StringVar()
        
        def select_method(method):
            method_var.set(method)
            dialog.destroy()
        
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Delete", command=lambda: select_method("delete")).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Zero", command=lambda: select_method("zero")).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Fill", command=lambda: select_method("fill")).pack(side=tk.LEFT, padx=5)
        
        dialog.wait_window()
        
        method = method_var.get()
        fill_val = None
        if method == "fill":
            fill_val = simpledialog.askstring("Fill Value", "Enter value to fill missing:")
        
        if method and (method != "fill" or fill_val is not None):
            new_df = handle_missing_values(ds.df, method=method, fill_value=fill_val)
            new_name = generate_temp_name(base="clean")
            self.manager.add_dataset(new_name, new_df, temporary=True)
            self.refresh_listbox()
            
            # Select the new dataset
            idx = list(self.manager.datasets.keys()).index(new_name)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self.listbox.activate(idx)
            self.manager.active_dataset_name = new_name
            
            messagebox.showinfo("Done", f"Missing values handled. New dataset '{new_name}' created.")
            self.update_undo_buttons()

    def standardize_data(self):
        ds = self.manager.get_active_dataset()
        if not ds:
            return
        
        # Create standardization dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Standardize Column")
        dialog.geometry("400x500")
        dialog.resizable(True, True)
        
        tk.Label(dialog, text="Select column to standardize:").pack(pady=5)
        
        column_var = tk.StringVar()
        column_menu = tk.OptionMenu(dialog, column_var, *ds.df.columns)
        column_menu.pack(pady=5)
        
        tk.Label(dialog, text="Select standardization method:").pack(pady=5)
        
        method_var = tk.StringVar()
        method_frame = tk.Frame(dialog)
        method_frame.pack(pady=5)
        
        # Additional input for methods that need it
        extra_frame = tk.Frame(dialog)
        extra_frame.pack(pady=5, fill=tk.X)
        extra_label = tk.Label(extra_frame, text="")
        extra_label.pack(side=tk.LEFT)
        extra_entry = tk.Entry(extra_frame, width=10)
        
        def on_column_change(*args):
            col = column_var.get()
            if col:
                dtype = str(ds.df[col].dtype)
                # Clear previous radiobuttons
                for widget in method_frame.winfo_children():
                    widget.destroy()
                if 'object' in dtype or 'string' in dtype:
                    methods = [
                        ('Convert to lowercase', 'lowercase'), 
                        ('Convert to uppercase', 'uppercase'), 
                        ('Convert to title case', 'title'), 
                        ('Strip whitespace', 'strip'),
                        ('Convert to number', 'words_to_num')
                    ]
                    extra_label.config(text="")
                    extra_entry.pack_forget()
                elif 'int' in dtype or 'float' in dtype:
                    methods = [
                        ('Round numbers', 'round'), 
                        ('Convert to numeric', 'to_numeric'),
                        ('Convert to words', 'num_to_words')
                    ]
                    extra_label.config(text="Number of decimal places:")
                    extra_entry.pack(side=tk.LEFT, padx=5)
                else:
                    methods = [('Convert to numeric', 'to_numeric')]  # fallback
                    extra_label.config(text="")
                    extra_entry.pack_forget()
                for text, val in methods:
                    tk.Radiobutton(method_frame, text=text, variable=method_var, value=val).pack(anchor=tk.W)
                method_var.set(methods[0][1] if methods else "")
        
        column_var.trace('w', on_column_change)
        
        # Set initial column if any
        if not ds.df.columns.empty:
            column_var.set(ds.df.columns[0])
            on_column_change()
        
        def apply_standardization():
            col = column_var.get()
            method = method_var.get()
            if not col or not method:
                messagebox.showwarning("Incomplete", "Select column and method")
                return
            kwargs = {}
            if method == 'round':
                try:
                    kwargs['decimals'] = int(extra_entry.get())
                except ValueError:
                    messagebox.showerror("Error", "Enter valid number for decimals")
                    return
            new_df = standardize_column(ds.df, col, method, **kwargs)
            new_name = generate_temp_name(base="std")
            self.manager.add_dataset(new_name, new_df, temporary=True)
            self.refresh_listbox()
            
            # Select the new dataset
            idx = list(self.manager.datasets.keys()).index(new_name)
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self.listbox.activate(idx)
            self.manager.active_dataset_name = new_name
            
            messagebox.showinfo("Done", f"Column '{col}' standardized. New dataset '{new_name}' created.")
            dialog.destroy()
            self.update_undo_buttons()
        
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=20)
        tk.Button(button_frame, text="Apply", command=apply_standardization).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=10)

    # ---------------- Cross-file ops ----------------
    def cross_file_merge(self):
        if len(self.selection_order) < 2:
            messagebox.showwarning("Selection Error", "Select at least two datasets")
            return
        names = self.selection_order
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
        top.title(f"Preview: {ds.name} | Shape: {ds.df.shape}")
        text = tk.Text(top, width=120, height=40)
        text.pack(expand=True, fill=tk.BOTH)
        
        # Show info and first 100 rows
        info = f"Total Rows: {len(ds.df)}\nTotal Columns: {len(ds.df.columns)}\n\n"
        text.insert(tk.END, info + ds.df.head(100).to_string())

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

    def update_undo_buttons(self):
        ds = self.manager.get_active_dataset()
        if ds:
            self.undo_btn.config(state=tk.NORMAL if len(ds.history) > 1 else tk.DISABLED)
            self.reset_btn.config(state=tk.NORMAL)
        else:
            self.undo_btn.config(state=tk.DISABLED)
            self.reset_btn.config(state=tk.DISABLED)
 