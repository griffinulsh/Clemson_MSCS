import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class Calculator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Scientific Calculator")
        self.root.configure(bg='#2C3E50')  # Dark blue background
        
        # Add padding around the main window
        self.root.geometry("1200x600")  # Made window wider for graph
        self.root.resizable(False, False)
        
        # Create left frame for calculator
        left_frame = tk.Frame(self.root, bg='#2C3E50', padx=20, pady=20)
        left_frame.pack(side=tk.LEFT, fill='both', expand=True)
        
        # Create right frame for graph
        self.right_frame = tk.Frame(self.root, bg='#2C3E50', padx=20, pady=20)
        self.right_frame.pack(side=tk.RIGHT, fill='both', expand=True)

        # Style constants
        button_style = {
            'font': ('Helvetica', 14, 'bold'),  # Made text bold
            'bg': '#E8E8E8',  # Light gray background
            'fg': '#2C3E50',  # Dark text
            'activebackground': '#D0D0D0',  # Slightly darker when clicked
            'activeforeground': '#2C3E50',
            'relief': 'flat',
            'width': 5,
            'height': 2,
            'bd': 0,
            'padx': 5,
            'pady': 5
        }
        
        operator_style = button_style.copy()
        operator_style['bg'] = '#FF9999'  # Light red
        operator_style['fg'] = '#8B0000'  # Dark red text
        operator_style['activebackground'] = '#FF7777'
        operator_style['activeforeground'] = '#8B0000'
        
        # Calculator display
        display_frame = tk.Frame(left_frame, bg='#2C3E50', pady=10)
        display_frame.grid(row=0, column=0, columnspan=4, sticky='nsew')
        
        self.entry = tk.Entry(display_frame, 
                            font=('Helvetica', 24),
                            bg='#ECF0F1',
                            fg='#2C3E50',
                            bd=0,
                            relief='flat',
                            justify='right',
                            width=20)
        self.entry.pack(pady=10, ipady=10)

        # Button layout
        self.buttons = [
            '7', '8', '9', '/',
            '4', '5', '6', '*',
            '1', '2', '3', '-',
            '0', '.', '=', '+'
        ]
        
        buttons_frame = tk.Frame(left_frame, bg='#2C3E50')
        buttons_frame.grid(row=1, column=0, columnspan=4, sticky='nsew')
        
        row_val = 0
        col_val = 0
        for button in self.buttons:
            style = operator_style if button in '/*-+=' else button_style
            btn = tk.Button(buttons_frame, 
                          text=button,
                          command=lambda button=button: self.click_button(button),
                          **style)
            btn.grid(row=row_val, column=col_val, padx=5, pady=5, sticky='nsew')
            col_val += 1
            if col_val > 3:
                col_val = 0
                row_val += 1

        # Clear button with special styling
        clear_style = button_style.copy()
        clear_style['bg'] = '#90EE90'  # Light green
        clear_style['fg'] = '#006400'  # Dark green text
        clear_style['activebackground'] = '#7CCD7C'
        clear_style['activeforeground'] = '#006400'
        self.clear_button = tk.Button(buttons_frame,
                                    text="Clear",
                                    command=self.clear,
                                    **clear_style)
        self.clear_button.grid(row=row_val, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')

        # Graphing section
        graph_frame = tk.Frame(left_frame, bg='#2C3E50', pady=10)
        graph_frame.grid(row=2, column=0, columnspan=4, sticky='nsew')
        
        # Graph input styling
        tk.Label(graph_frame,
                text="Enter equation (e.g., y = 2x + 3):",
                font=('Helvetica', 12),
                bg='#2C3E50',
                fg='white').pack(pady=(10,5))
        
        self.graph_entry = tk.Entry(graph_frame,
                                  font=('Helvetica', 14),
                                  bg='#ECF0F1',
                                  fg='#2C3E50',
                                  bd=0,
                                  relief='flat',
                                  width=20)
        self.graph_entry.pack(pady=5, ipady=5)
        self.graph_entry.insert(0, "y = x")
        
        # Graph button styling
        graph_style = button_style.copy()
        graph_style['bg'] = '#ADD8E6'  # Light blue
        graph_style['fg'] = '#00008B'  # Dark blue text
        graph_style['activebackground'] = '#87CEEB'
        graph_style['activeforeground'] = '#00008B'
        self.graph_button = tk.Button(graph_frame,
                                    text="Graph",
                                    command=self.plot_graph,
                                    **graph_style)
        self.graph_button.pack(pady=10)

        # Configure plot style
        plt_style = {
            'figure.facecolor': '#ECF0F1',
            'axes.facecolor': '#ECF0F1',
            'axes.edgecolor': '#2C3E50',
            'axes.labelcolor': '#2C3E50',
            'xtick.color': '#2C3E50',
            'ytick.color': '#2C3E50',
            'grid.color': '#BDC3C7'
        }

    def clear(self):
        self.entry.delete(0, tk.END)

    def click_button(self, button):
        current = self.entry.get()
        if button == '=':
            try:
                result = str(eval(current))
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, result)
            except:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, "Error")
        else:
            self.entry.insert(tk.END, button)

    def plot_graph(self):
        try:
            equation = self.graph_entry.get().replace(' ', '')
            
            if '=' in equation:
                equation = equation.split('=')[1]
            
            x = np.linspace(-10, 10, 100)
            equation = equation.replace('x', '*x')
            y = eval(equation)
            
            # Clear previous graph
            for widget in self.right_frame.winfo_children():
                widget.destroy()
            
            fig = Figure(figsize=(6, 5), dpi=100)
            fig.patch.set_facecolor('#ECF0F1')
            
            ax = fig.add_subplot(111)
            ax.set_facecolor('#ECF0F1')
            ax.plot(x, y, color='#E74C3C', linewidth=2)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.axhline(y=0, color='#2C3E50', linestyle='-', linewidth=0.5)
            ax.axvline(x=0, color='#2C3E50', linestyle='-', linewidth=0.5)
            
            # Style the axes
            ax.spines['left'].set_color('#2C3E50')
            ax.spines['bottom'].set_color('#2C3E50')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            self.canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            print(f"Error plotting graph: {str(e)}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    calc = Calculator()
    calc.run()

