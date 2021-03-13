from tkinter import *
import tkinter.messagebox
from tkinter import filedialog
from tkinter import simpledialog
from PIL import ImageTk, Image, ImageDraw
import PIL

from detect_symbols import SymbolDetector


class DrawPad:
    def __init__(self):
        self.drawing_color = '#007acc'  # Could be expressed with Hex color codes
        self.background_color = 'white'
        self.width = 800
        self.height = 500
        self.old_x = None
        self.old_y = None
        self.penWidth = 2
        self.root_window = Tk()  # Create the root Window
        self.root_window.geometry(f'{self.width}x{self.height}')  # Set the size of the root window
        self.symbol_detector = SymbolDetector("formula.png", (-1*self.penWidth + 6, -1*self.penWidth + 6))  # kernel_size = -1*penWidth + 6
        self._create_buttons_frame()
        self._create_canvas()
        self._create_memory_image()

    def _create_canvas(self):
        self.canvas = Canvas(self.root_window, width=self.width, height=self.height, bg=self.background_color)  # Create the Canvas Window
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind('<B1-Motion>', self.draw_line)  # Draw the line when the mouse button is pressed and the mouse is moved
        self.canvas.bind('<ButtonRelease-1>', self.reset)  # Reset the coordinates when the mouse button is released

    def _create_memory_image(self):
        self.memory_image = PIL.Image.new("RGB", (self.width, self.height), (255, 255, 255))  # Creat a memory image to save the drawing
        self.memory_draw = ImageDraw.Draw(self.memory_image)

    def detect(self):
        self.symbol_detector.detect_symbols()

    def _create_buttons_frame(self):
        self.detect_btn = Button(self.root_window, text="Detect", command=lambda: self.symbol_detector.detect_symbols())
        self.detect_btn.pack(side=tkinter.BOTTOM)

    def draw_line(self, e):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, e.x, e.y, width=self.penWidth, fill=self.drawing_color,
                                    capstyle=ROUND, smooth=True)
            self.memory_draw.line([self.old_x, self.old_y, e.x, e.y], width=self.penWidth, fill=self.drawing_color)
        self.old_x = e.x
        self.old_y = e.y
        self.save_drawing()

    def save_drawing(self):
        filename = "formula.png"
        self.memory_image.save(filename)

    def reset(self, e):  # Resetting or cleaning the canvas
        self.old_x = None
        self.old_y = None

    def clear(self):
        self.canvas.delete('all')
        self.memory_draw.rectangle((0, 0, 500, 500), fill=(255, 255, 255, 0))


# if __name__ == '__main__':
#     drawPad = DrawPad()
#     drawPad.root_window.title("Draw Pad ✏️")
#     drawPad.root_window.mainloop()
