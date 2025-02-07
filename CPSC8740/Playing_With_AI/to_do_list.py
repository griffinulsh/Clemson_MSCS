import tkinter as tk
from tkinter import ttk, messagebox
import json
from pathlib import Path

class TodoList:
    def __init__(self):
        self.tasks = []
        self.load_tasks()

    def add_task(self, task):
        self.tasks.append({"task": task, "done": False})
        self.save_tasks()

    def view_tasks(self):
        return self.tasks

    def delete_task(self, task_number):
        try:
            del self.tasks[task_number]
            self.save_tasks()
            return True
        except IndexError:
            return False

    def mark_done(self, task_number):
        try:
            self.tasks[task_number]["done"] = not self.tasks[task_number]["done"]
            self.save_tasks()
            return True
        except IndexError:
            return False

    def save_tasks(self):
        with open('tasks.json', 'w') as f:
            json.dump(self.tasks, f)

    def load_tasks(self):
        try:
            with open('tasks.json', 'r') as f:
                self.tasks = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.tasks = []

class TodoApp:
    def __init__(self, root):
        self.root = root
        self.todo = TodoList()
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Todo List")
        self.root.geometry("400x600")
        self.root.configure(bg="#f0f0f0")

        style = ttk.Style()
        style.configure("Custom.TFrame", background="#f0f0f0")
        style.configure("Custom.TButton", padding=5)
        style.configure("Task.TFrame", background="white")

        # Main container
        main_frame = ttk.Frame(self.root, style="Custom.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Task input area
        input_frame = ttk.Frame(main_frame, style="Custom.TFrame")
        input_frame.pack(fill=tk.X, pady=(0, 10))

        self.task_entry = ttk.Entry(input_frame)
        self.task_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.task_entry.bind("<Return>", lambda e: self.add_task())

        add_button = ttk.Button(input_frame, text="Add Task", command=self.add_task)
        add_button.pack(side=tk.RIGHT)

        # Tasks list area
        self.tasks_canvas = tk.Canvas(main_frame, bg="white")
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.tasks_canvas.yview)
        self.tasks_frame = ttk.Frame(self.tasks_canvas, style="Custom.TFrame")

        self.tasks_frame.bind("<Configure>", lambda e: self.tasks_canvas.configure(scrollregion=self.tasks_canvas.bbox("all")))
        self.tasks_canvas.create_window((0, 0), window=self.tasks_frame, anchor="nw", width=365)
        self.tasks_canvas.configure(yscrollcommand=scrollbar.set)

        self.tasks_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.update_task_list()

    def add_task(self):
        task_text = self.task_entry.get().strip()
        if task_text:
            self.todo.add_task(task_text)
            self.task_entry.delete(0, tk.END)
            self.update_task_list()

    def delete_task(self, index):
        task = self.todo.view_tasks()[index]
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the task:\n'{task['task']}'?"):
            if self.todo.delete_task(index):
                self.update_task_list()

    def toggle_task(self, index):
        if self.todo.mark_done(index):
            self.update_task_list()

    def update_task_list(self):
        # Clear existing tasks
        for widget in self.tasks_frame.winfo_children():
            widget.destroy()

        # Add tasks
        for i, task in enumerate(self.todo.view_tasks()):
            task_frame = ttk.Frame(self.tasks_frame, style="Task.TFrame")
            task_frame.pack(fill=tk.X, pady=2)

            check_var = tk.BooleanVar(value=task["done"])
            checkbox = ttk.Checkbutton(task_frame, variable=check_var, 
                                     command=lambda i=i: self.toggle_task(i))
            checkbox.pack(side=tk.LEFT)
            
            if task["done"]:
                label = ttk.Label(task_frame, text=task["task"], 
                                foreground="gray", font=("TkDefaultFont", 10, "overstrike"))
            else:
                label = ttk.Label(task_frame, text=task["task"])
            label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            delete_btn = ttk.Button(task_frame, text="×", width=3,
                                  command=lambda i=i: self.delete_task(i))
            delete_btn.pack(side=tk.RIGHT)

def main():
    root = tk.Tk()
    app = TodoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
