import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog, Menu
from core.state import DatasetManager
from core.loader import load_file
from core.processor import remove_duplicates, handle_missing_values, standardize_data, standardize_column, merge_datasets
from core.utils import generate_temp_name
from nlp.engine import NLPEngine

# Set the Space Theme palette globally
ctk.set_appearance_mode("Dark")  

# Nasa Space Apps Inspired Palette constants
SPACE_VOID = "#0c0c0c"
SPACE_MIDNIGHT = "#1a1a2e"
SPACE_DEEP_BLUE = "#16213e"
NEON_CYAN = "#00f3ff"
NEON_CYAN_HOVER = "#00c8d1"
NEON_PURPLE = "#b026ff"
NEON_PURPLE_HOVER = "#8f12d6"
NEON_GREEN = "#39ff14"
TEXT_GLOW = "#e2e8f0"

class DataProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPACE DATA TERMINAL 🚀")
        self.root.geometry("1000x750")
        self.root.minsize(900, 650)
        self.root.configure(bg=SPACE_VOID)
        
        self.manager = DatasetManager()
        self.nlp = NLPEngine()
        self.selection_order = []  
        self.create_widgets()

    def create_widgets(self):
        # Master Layout: Left panel for controls, Right panel for listbox and NLP
        
        # Left Panel (Controls)
        self.left_panel = ctk.CTkFrame(self.root, corner_radius=15, fg_color=SPACE_MIDNIGHT, border_width=1, border_color=SPACE_DEEP_BLUE)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=20, pady=20)
        
        title_label = ctk.CTkLabel(self.left_panel, text="Preprocessing Magic", font=ctk.CTkFont(family="Orbitron", size=22, weight="bold"), text_color=NEON_CYAN)
        title_label.pack(pady=(20, 20), padx=20)

        # File Management
        upload_btn = ctk.CTkButton(self.left_panel, text="📂 Upload Datasets", command=self.upload_files, font=ctk.CTkFont(size=14, weight="bold"), height=40, fg_color=NEON_PURPLE, hover_color=NEON_PURPLE_HOVER)
        upload_btn.pack(fill=tk.X, padx=20, pady=(0, 20))

        # Data Cleaning
        clean_lbl = ctk.CTkLabel(self.left_panel, text="🧹 Cleaning Actions", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_GLOW)
        clean_lbl.pack(anchor="w", padx=20, pady=(10, 5))

        self.remove_dup_btn = ctk.CTkButton(self.left_panel, text="Remove Duplicates", command=self.remove_duplicates, state="disabled", fg_color="transparent", border_width=2, border_color=NEON_CYAN, text_color=NEON_CYAN, hover_color=SPACE_DEEP_BLUE)
        self.remove_dup_btn.pack(fill=tk.X, padx=20, pady=5)
        
        self.handle_missing_btn = ctk.CTkButton(self.left_panel, text="Handle Missing", command=self.handle_missing_values, state="disabled", fg_color="transparent", border_width=2, border_color=NEON_CYAN, text_color=NEON_CYAN, hover_color=SPACE_DEEP_BLUE)
        self.handle_missing_btn.pack(fill=tk.X, padx=20, pady=5)
        
        self.standardize_btn = ctk.CTkButton(self.left_panel, text="Standardize Data", command=self.standardize_data, state="disabled", fg_color="transparent", border_width=2, border_color=NEON_CYAN, text_color=NEON_CYAN, hover_color=SPACE_DEEP_BLUE)
        self.standardize_btn.pack(fill=tk.X, padx=20, pady=5)

        self.auto_clean_btn = ctk.CTkButton(self.left_panel, text="✨ Auto-Clean (Semantic)", command=self.auto_clean, state="disabled", fg_color="transparent", border_width=2, border_color=NEON_GREEN, text_color=NEON_GREEN, hover_color=SPACE_DEEP_BLUE)
        self.auto_clean_btn.pack(fill=tk.X, padx=20, pady=5)

        # Advanced
        adv_lbl = ctk.CTkLabel(self.left_panel, text="🔗 Advanced", font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_GLOW)
        adv_lbl.pack(anchor="w", padx=20, pady=(15, 5))

        self.merge_btn = ctk.CTkButton(self.left_panel, text="Merge Datasets", command=self.cross_file_merge, state="disabled", fg_color="transparent", border_width=2, border_color=NEON_PURPLE, text_color=NEON_PURPLE, hover_color=SPACE_DEEP_BLUE)
        self.merge_btn.pack(fill=tk.X, padx=20, pady=5)

        # Right Panel (Listbox, Preview, NLP)
        self.right_panel = ctk.CTkFrame(self.root, fg_color="transparent")
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20), pady=20)

        # Top Right: Dataset List
        list_frame = ctk.CTkFrame(self.right_panel, corner_radius=15, fg_color=SPACE_MIDNIGHT, border_width=1, border_color=SPACE_DEEP_BLUE)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        list_lbl = ctk.CTkLabel(list_frame, text="🗂️ Active Datasets", font=ctk.CTkFont(size=16, weight="bold"), text_color=TEXT_GLOW)
        list_lbl.pack(anchor="w", padx=20, pady=(15, 5))

        # We style a basic Tkinter listbox to match CustomTkinter
        self.listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, bg=SPACE_DEEP_BLUE, fg=TEXT_GLOW, selectbackground=NEON_PURPLE, selectforeground="white", font=("Orbitron", 12), borderwidth=0, highlightthickness=0)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=(5, 20))
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.listbox.bind("<Button-3>", self.show_context_menu)

        # Actions Row (Undo, Reset, View, Export)
        action_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        action_frame.pack(fill=tk.X, pady=(0, 20))

        self.preview_btn = ctk.CTkButton(action_frame, text="👁️ Preview", command=self.preview_data, state="disabled", width=100, fg_color=NEON_GREEN, text_color="black", hover_color="#2dd412")
        self.preview_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.export_btn = ctk.CTkButton(action_frame, text="💾 Export", command=self.export_dataset, state="disabled", width=100, fg_color="transparent", border_width=2, border_color=NEON_GREEN, text_color=NEON_GREEN, hover_color=SPACE_DEEP_BLUE)
        self.export_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.pipeline_btn = ctk.CTkButton(action_frame, text="📜 Pipeline", command=self.view_pipeline, state="disabled", width=100, fg_color="transparent", border_width=2, border_color=NEON_PURPLE, text_color=NEON_PURPLE, hover_color=SPACE_DEEP_BLUE)
        self.pipeline_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.reset_btn = ctk.CTkButton(action_frame, text="🔄 Reset", command=self.reset, state="disabled", width=90, fg_color="transparent", border_width=1, border_color=TEXT_GLOW, hover_color=SPACE_DEEP_BLUE)
        self.reset_btn.pack(side=tk.RIGHT, padx=(5, 0))

        self.undo_btn = ctk.CTkButton(action_frame, text="↶ Undo", command=self.undo, state="disabled", width=90, fg_color="transparent", border_width=1, border_color=TEXT_GLOW, hover_color=SPACE_DEEP_BLUE)
        self.undo_btn.pack(side=tk.RIGHT)

        # Bottom Right: NLP Command Line
        nlp_frame = ctk.CTkFrame(self.right_panel, corner_radius=15, fg_color=SPACE_MIDNIGHT, border_width=2, border_color=NEON_CYAN)
        nlp_frame.pack(fill=tk.X)
        
        nlp_lbl = ctk.CTkLabel(nlp_frame, text="✨ Magical AI Command Line", font=ctk.CTkFont(family="Orbitron", size=14, weight="bold"), text_color=NEON_CYAN)
        nlp_lbl.pack(anchor="w", padx=20, pady=(15, 5))

        input_row = ctk.CTkFrame(nlp_frame, fg_color="transparent")
        input_row.pack(fill=tk.X, padx=20, pady=(0, 20))

        self.nlp_input_var = tk.StringVar()
        self.nlp_entry = ctk.CTkEntry(input_row, textvariable=self.nlp_input_var, font=ctk.CTkFont(size=14), placeholder_text="e.g. 'remove duplicates'", height=40, state="disabled", fg_color=SPACE_DEEP_BLUE, border_color=SPACE_DEEP_BLUE)
        self.nlp_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
        self.nlp_entry.bind('<Return>', lambda e: self.execute_nlp_command())
        
        self.nlp_btn = ctk.CTkButton(input_row, text="🚀 Execute", command=self.execute_nlp_command, state="disabled", width=100, height=40, fg_color=NEON_CYAN, text_color="black", hover_color=NEON_CYAN_HOVER)
        self.nlp_btn.pack(side=tk.RIGHT)

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
        self.selection_order = []  

    def on_select(self, event):
        current_selection = set(self.listbox.curselection())
        previous_selection = set(self.selection_order)
        
        newly_selected = current_selection - previous_selection
        newly_deselected = previous_selection - current_selection
        
        for name in newly_deselected:
            if name in self.selection_order:
                self.selection_order.remove(name)
        
        for idx in newly_selected:
            name = self.listbox.get(idx)
            if name not in self.selection_order:
                self.selection_order.append(name)
        
        selection = self.listbox.curselection()
        if selection:
            self.manager.active_dataset_name = self.selection_order[0] if self.selection_order else None
            
            # Enable buttons using CTk state configuration
            for btn in [self.remove_dup_btn, self.handle_missing_btn, self.standardize_btn, self.preview_btn, self.export_btn, self.pipeline_btn, self.auto_clean_btn]:
                btn.configure(state="normal")
                
            ds = self.manager.get_active_dataset()
            if ds:
                self.reset_btn.configure(state="normal")
                self.undo_btn.configure(state="normal" if len(ds.history) > 1 else "disabled")
                
            if len(self.selection_order) >= 2:
                self.merge_btn.configure(state="normal")
            else:
                self.merge_btn.configure(state="disabled")
                
            self.nlp_entry.configure(state="normal")
            self.nlp_btn.configure(state="normal")
        else:
            self.manager.active_dataset_name = None
            self.selection_order = []
            for btn in [self.remove_dup_btn, self.handle_missing_btn, self.standardize_btn, self.preview_btn, self.export_btn, self.merge_btn, self.undo_btn, self.reset_btn, self.nlp_btn, self.pipeline_btn, self.auto_clean_btn]:
                btn.configure(state="disabled")
            self.nlp_entry.configure(state="disabled")

    def show_context_menu(self, event):
        index = self.listbox.nearest(event.y)
        if index < 0 or index >= self.listbox.size():
            return
        
        name = self.listbox.get(index)
        if name not in self.manager.datasets:
            return
        
        menu = Menu(self.root, tearoff=0, bg=SPACE_MIDNIGHT, fg="white", activebackground=NEON_PURPLE)
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
            self.on_select(None)  
        self.update_undo_buttons()

    def update_undo_buttons(self):
        ds = self.manager.get_active_dataset()
        if ds:
            self.undo_btn.configure(state="normal" if len(ds.history) > 1 else "disabled")
            self.reset_btn.configure(state="normal")
        else:
            self.undo_btn.configure(state="disabled")
            self.reset_btn.configure(state="disabled")

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

    def auto_clean(self):
        ds = self.manager.get_active_dataset()
        if not ds: return
        
        logs = ds.auto_clean()
        self.update_undo_buttons()
        
        if logs:
            msg = "Semantic Type Detection found errors:\n\n" + "\n".join(logs)
            messagebox.showinfo("Auto-Clean Complete", msg)
        else:
            messagebox.showinfo("Auto-Clean Complete", "Your dataset looks perfectly clean semantically! ✨")

    def remove_duplicates(self):
        ds = self.manager.get_active_dataset()
        if ds:
            new_df = remove_duplicates(ds.df)
            new_name = generate_temp_name(base="deduped")
            self.manager.add_dataset(new_name, new_df, temporary=True)
            self.refresh_listbox()
            self._select_new_dataset(new_name)
            messagebox.showinfo("Done", f"Duplicates removed. New dataset '{new_name}' created.")
            self.update_undo_buttons()

    def handle_missing_values(self):
        ds = self.manager.get_active_dataset()
        if not ds: return
        
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Handle Missing Values")
        dialog.geometry("350x200")
        dialog.resizable(False, False)
        dialog.grab_set()
        
        dialog.configure(fg_color=SPACE_VOID)
        ctk.CTkLabel(dialog, text="Choose method:", font=ctk.CTkFont(weight="bold"), text_color=TEXT_GLOW).pack(pady=15)
        
        method_var = tk.StringVar()
        
        def select_method(method):
            method_var.set(method)
            dialog.destroy()
        
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(pady=10)
        
        ctk.CTkButton(button_frame, text="Delete/Drop", command=lambda: select_method("delete"), width=90, fg_color="transparent", border_width=2, border_color="#ff2a2a", text_color="#ff2a2a", hover_color=SPACE_DEEP_BLUE).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(button_frame, text="Zero Out", command=lambda: select_method("zero"), width=90, fg_color="transparent", border_width=2, border_color=NEON_CYAN, text_color=NEON_CYAN, hover_color=SPACE_DEEP_BLUE).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(button_frame, text="Custom Fill", command=lambda: select_method("fill"), width=90, fg_color="transparent", border_width=2, border_color=NEON_GREEN, text_color=NEON_GREEN, hover_color=SPACE_DEEP_BLUE).pack(side=tk.LEFT, padx=5)
        
        self.root.wait_window(dialog)
        
        method = method_var.get()
        fill_val = None
        if method == "fill":
            fill_val = simpledialog.askstring("Fill Value", "Enter value to fill missing:")
        
        if method and (method != "fill" or fill_val is not None):
            new_df = handle_missing_values(ds.df, method=method, fill_value=fill_val)
            new_name = generate_temp_name(base="clean")
            self.manager.add_dataset(new_name, new_df, temporary=True)
            self.refresh_listbox()
            self._select_new_dataset(new_name)
            messagebox.showinfo("Done", f"Missing values handled. New dataset '{new_name}' created.")
            self.update_undo_buttons()

    def standardize_data(self):
        ds = self.manager.get_active_dataset()
        if not ds: return
        
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Standardize Column")
        dialog.geometry("450x550")
        dialog.grab_set()
        dialog.configure(fg_color=SPACE_VOID)
        
        ctk.CTkLabel(dialog, text="Select column:", font=ctk.CTkFont(weight="bold"), text_color=TEXT_GLOW).pack(pady=(15, 5))
        
        cols = list(ds.df.columns)
        column_var = tk.StringVar(value=cols[0] if cols else "")
        column_menu = ctk.CTkOptionMenu(dialog, variable=column_var, values=cols, fg_color=SPACE_DEEP_BLUE, button_color=SPACE_DEEP_BLUE, button_hover_color=NEON_PURPLE)
        column_menu.pack(pady=5)
        
        ctk.CTkLabel(dialog, text="Select method:", font=ctk.CTkFont(weight="bold"), text_color=TEXT_GLOW).pack(pady=(15, 5))
        
        method_var = tk.StringVar()
        method_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        method_frame.pack(pady=5, fill=tk.BOTH, expand=True, padx=20)
        
        extra_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        extra_frame.pack(pady=10, fill=tk.X, padx=20)
        extra_label = ctk.CTkLabel(extra_frame, text="")
        extra_label.pack(side=tk.LEFT)
        extra_entry = ctk.CTkEntry(extra_frame, width=80)
        
        radio_buttons = []

        def on_column_change(*args):
            col = column_var.get()
            if not col: return
            
            dtype = str(ds.df[col].dtype)
            for widget in radio_buttons:
                widget.destroy()
            radio_buttons.clear()

            if 'object' in dtype or 'string' in dtype:
                methods = [
                    ('Convert to lowercase', 'lowercase'), 
                    ('Convert to uppercase', 'uppercase'), 
                    ('Convert to title case', 'title'), 
                    ('Strip whitespace', 'strip'),
                    ('Convert to number', 'words_to_num')
                ]
                sample = ds.df[col].dropna().iloc[0] if not ds.df[col].dropna().empty else ""
                sample_str = str(sample)
                
                if "date" in col.lower() or "time" in col.lower() or any(x in sample_str for x in ['/', '-']):
                    methods.append(('Standardize Date (YYYY-MM-DD)', 'to_date'))
                if "price" in col.lower() or "amount" in col.lower() or "$" in sample_str:
                    methods.append(('Remove Currency Symbols', 'remove_currency'))

                extra_label.configure(text="")
                extra_entry.pack_forget()
            elif 'int' in dtype or 'float' in dtype:
                methods = [
                    ('Round numbers', 'round'),
                    ('Convert to words', 'num_to_words')
                ]
                extra_label.configure(text="Number of decimals:")
                extra_entry.pack(side=tk.LEFT, padx=10)
            else:
                methods = [('Convert to numeric', 'to_numeric')] 
                extra_label.configure(text="")
                extra_entry.pack_forget()
            
            for text, val in methods:
                rb = ctk.CTkRadioButton(method_frame, text=text, variable=method_var, value=val, text_color=TEXT_GLOW, hover_color=NEON_PURPLE)
                rb.pack(anchor="w", pady=5)
                radio_buttons.append(rb)
            
            if methods:
                method_var.set(methods[0][1])
                
        column_var.trace('w', on_column_change)
        on_column_change()
        
        def apply_standardization():
            col = column_var.get()
            method = method_var.get()
            if not col or not method:
                messagebox.showwarning("Incomplete", "Select method")
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
            self._select_new_dataset(new_name)
            messagebox.showinfo("Done", f"Column standardized. New dataset '{new_name}' created.")
            dialog.destroy()
            self.update_undo_buttons()
        
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(pady=(0, 20))
        ctk.CTkButton(btn_frame, text="Apply", command=apply_standardization, width=100, fg_color=NEON_GREEN, text_color="black", hover_color="#2dd412").pack(side=tk.LEFT, padx=10)
        ctk.CTkButton(btn_frame, text="Cancel", command=dialog.destroy, width=100, fg_color="transparent", border_width=1, border_color=TEXT_GLOW, text_color=TEXT_GLOW, hover_color=SPACE_DEEP_BLUE).pack(side=tk.LEFT, padx=10)

    def cross_file_merge(self):
        if len(self.selection_order) < 2:
            messagebox.showwarning("Selection Error", "Select at least two datasets")
            return
            
        try:
            dfs_to_merge = [self.manager.datasets[name].df for name in self.selection_order if name in self.manager.datasets]
            new_df, stats = self.manager.forge.vertical_concatenation(dfs_to_merge)
            temp_name = generate_temp_name("forged")
            self.manager.add_dataset(temp_name, new_df, temporary=True)
            self.manager.datasets[temp_name].save_state(operation_name="Data Forge: Multi-Merge", stats=stats)
            messagebox.showinfo("Forge Success", f"Successfully merged {stats['dataframe_count']} datasets into {temp_name}!")
            self.refresh_listbox()
            self._select_new_dataset(temp_name)
        except Exception as e:
            messagebox.showerror("Merge Error", f"Failed to forge datasets: {str(e)}")

    def view_pipeline(self):
        ds = self.manager.get_active_dataset()
        if not ds: return
        
        tracker_history = ds.tracker.get_history()
        
        dialog = ctk.CTkToplevel(self.root, fg_color=SPACE_VOID)
        dialog.title(f"Pipeline Audit Log: {ds.name}")
        dialog.geometry("600x400")
        dialog.grab_set()

        lbl = ctk.CTkLabel(dialog, text=f"Pipeline History: {ds.name}", font=ctk.CTkFont(family="Orbitron", size=18, weight="bold"), text_color=NEON_PURPLE)
        lbl.pack(pady=10)
        
        text_area = ctk.CTkTextbox(dialog, width=560, height=300, fg_color=SPACE_MIDNIGHT, text_color=TEXT_GLOW, font=("Consolas", 12))
        text_area.pack(pady=10)
        
        for i, step in enumerate(tracker_history):
            text_area.insert("end", f"[{i+1}] {step['timestamp']}\n")
            text_area.insert("end", f"Operation: {step['operation']}\n")
            text_area.insert("end", f"Stats: {step['stats']}\n")
            text_area.insert("end", "-"*40 + "\n")
        
        text_area.configure(state="disabled")

    def preview_data(self):
        ds = self.manager.get_active_dataset()
        if not ds: return
        
        top = ctk.CTkToplevel(self.root)
        top.title(f"Preview: {ds.name} | Shape: {ds.df.shape}")
        top.geometry("800x600")
        
        textbox = ctk.CTkTextbox(top, font=ctk.CTkFont(family="Courier", size=12), wrap="none")
        textbox.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        
        info = f"Total Rows: {len(ds.df)}\nTotal Columns: {len(ds.df.columns)}\n\n"
        textbox.insert("1.0", info + ds.df.head(100).to_string())
        textbox.configure(state="disabled")

    def export_dataset(self):
        ds = self.manager.get_active_dataset()
        if not ds: return
        filetypes = [("CSV", "*.csv"), ("Excel", "*.xlsx"), ("JSON", "*.json")]
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=filetypes)
        if not save_path: return

        try:
            if save_path.endswith(".csv"):
                ds.df.to_csv(save_path, index=False)
            elif save_path.endswith(".xlsx"):
                ds.df.to_excel(save_path, index=False)
            elif save_path.endswith(".json"):
                ds.df.to_json(save_path, orient="records")
            else:
                messagebox.showerror("Error", "Unsupported format")
                return
            messagebox.showinfo("Exported", f"Dataset saved as {save_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def execute_nlp_command(self):
        command = self.nlp_input_var.get().strip()
        if not command: return
            
        ds = self.manager.get_active_dataset()
        if not ds:
            messagebox.showwarning("No Dataset", "Please select a dataset first.")
            return

        result = self.nlp.parse_command(command, ds.df)
        if not result:
            messagebox.showinfo("NLP Engine", f"Sorry, I didn't understand:\n'{command}'")
            return
            
        intent = result["intent"]
        
        if intent == "advanced_operation":
            op = result["operation"]
            details = result["details"]
            try:
                new_df = self.nlp.execute_advanced(ds.df, details)
                new_name = generate_temp_name(base=f"nlp_{op}")
                self.manager.add_dataset(new_name, new_df, temporary=True)
                self.manager.get_active_dataset().save_state(operation_name=f"NLP: {details.get('description', op)}")
                self.refresh_listbox()
                self._select_new_dataset(new_name)
                messagebox.showinfo("NLP Success", f"Successfully executed: {details.get('description', op)} ✨")
            except Exception as e:
                messagebox.showerror("NLP Error", f"Failed to execute {op}: {str(e)}")
        elif intent == "remove_duplicates":
            self.remove_duplicates()
            messagebox.showinfo("NLP Success", f"Magically {intent.replace('_', ' ')}! ✨")
        elif intent == "drop_missing":
            new_df = handle_missing_values(ds.df, method="delete")
            new_name = generate_temp_name(base="clean")
            self.manager.add_dataset(new_name, new_df, temporary=True)
            self.refresh_listbox()
            self._select_new_dataset(new_name)
            messagebox.showinfo("NLP Success", "Magically dropped missing values! ✨")
        elif intent == "fill_zero_missing":
            new_df = handle_missing_values(ds.df, method="zero")
            new_name = generate_temp_name(base="clean")
            self.manager.add_dataset(new_name, new_df, temporary=True)
            self.refresh_listbox()
            self._select_new_dataset(new_name)
            messagebox.showinfo("NLP Success", "Filled missing values with zeros! ✨")
        elif intent.startswith("standardize_") or intent == "remove_currency":
            messagebox.showinfo("NLP Suggestion", "Select column below.")
            self.standardize_data()
            
        self.nlp_input_var.set("")
        self.update_undo_buttons()
        
    def _select_new_dataset(self, new_name):
        idx = list(self.manager.datasets.keys()).index(new_name)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.activate(idx)
        self.manager.active_dataset_name = new_name
        self.on_select(None)
