#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plant_segmenter.py  (app/ entry point)
=======================================
Thin launcher — just starts the GUI.

Run from the app/ directory:
    python plant_segmenter.py

Or from the project root:
    python app/plant_segmenter.py
"""

import tkinter as tk
from gui_app import LeafSegmenterGUI


def main() -> None:
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.3)
    except Exception:
        pass
    _app = LeafSegmenterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
