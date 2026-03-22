"""
Translation Overlay Tool
========================
A always-on-top floating overlay that continuously captures the screen with
PaddleOCR every second and displays the translation.

Features:
  - Always on top of all windows
  - Drag to reposition anywhere on screen
  - PaddleOCR screen capture every 1 second
  - Auto-translate extracted text via Google Translate
  - Hotkey Ctrl+Shift+H to show/hide

Usage:
  python translation_overlay.py

Requirements:
  pip install keyboard Pillow paddleocr deep-translator numpy
"""

import os
import tkinter as tk
import threading
import time
import re

# Limit ONNX/numpy OpenMP thread pool before those libraries load.
# Without this, RapidOCR spawns one thread per CPU core on every inference.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

try:
    from PIL import ImageGrab, Image
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from rapidocr_onnxruntime import RapidOCR
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATE = True
except ImportError:
    HAS_TRANSLATE = False

# ── Configuration ──────────────────────────────────────────────────────────────
BG_COLOR        = "#1a1a2e"       # Dark navy background
TEXT_COLOR      = "#e0e0ff"       # Soft lavender-white text
ACCENT_COLOR    = "#7c6af7"       # Purple accent
BORDER_COLOR    = "#3a3a6e"       # Subtle border
ALPHA           = 0.85            # Window transparency (0.0 – 1.0)
FONT_FAMILY     = "Consolas"      # Monospace for clean look
FONT_SIZE       = 13
WIN_WIDTH       = 500
WIN_HEIGHT      = 320
HOTKEY          = "ctrl+shift+h"  # Toggle show/hide
OCR_LANG        = "en"         # PaddleOCR language: 'japan', 'ch', 'en', etc.
TRANSLATE_FROM  = "auto"          # Source language (auto-detect)
TRANSLATE_TO    = "zh-CN"         # Target translation language (zh-CN = Simplified Chinese)
OCR_INTERVAL    = 2.0             # Seconds between screen captures
MAX_OCR_WIDTH   = 1280            # Resize captured image to this width before OCR
# ───────────────────────────────────────────────────────────────────────────────


class TranslationOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.visible = True
        self._drag_x = 0
        self._drag_y = 0
        self._ocr_running = False
        self._ocr_thread = None
        self._paddle = None       # RapidOCR instance, initialized on first use
        self._capture_region = None  # None = full screen; (x1,y1,x2,y2) when set
        self._dpi_scale = None    # computed once on first region select
        self._word_cache = {}     # {word: translated} to avoid re-translating
        self._word_panel = None
        self._source_lang = "en"  # active input language for translation
        self._word_gen = 0        # incremented each time the word list changes
        self._last_img_hash = None   # thumbnail hash of previous captured frame
        self._last_raw_text = None   # OCR output of previous cycle
        self._mode = "auto"          # "auto" = continuous loop, "manual" = one-shot
        self._translator = None      # cached GoogleTranslator instance
        self._no_change_streak = 0   # consecutive idle cycles for adaptive back-off

        self._build_window()
        self._build_ui()
        self._bind_drag()
        self._register_hotkey()

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - WIN_WIDTH) // 2
        y = sh - WIN_HEIGHT - 80
        self.root.geometry(f"{WIN_WIDTH}x{WIN_HEIGHT}+{x}+{y}")
        self.root.update_idletasks()
        self._build_word_panel()

    # ── Window setup ────────────────────────────────────────────────────────────
    def _build_window(self):
        self.root.title("Translation Overlay")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", ALPHA)
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

    # ── UI construction ─────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = tk.Frame(self.root, bg=BORDER_COLOR, padx=1, pady=1)
        outer.pack(fill="both", expand=True)

        inner = tk.Frame(outer, bg=BG_COLOR)
        inner.pack(fill="both", expand=True)

        # ── Title bar ──
        title_bar = tk.Frame(inner, bg=BG_COLOR, height=28)
        title_bar.pack(fill="x", padx=6, pady=(6, 0))
        title_bar.pack_propagate(False)

        grip_lbl = tk.Label(
            title_bar, text="⠿ TRANSLATION OVERLAY",
            bg=BG_COLOR, fg=ACCENT_COLOR,
            font=(FONT_FAMILY, 8, "bold")
        )
        grip_lbl.pack(side="left", padx=(2, 0))

        hint = tk.Label(
            title_bar, text=f"[{HOTKEY.upper()}] hide",
            bg=BG_COLOR, fg="#555577",
            font=(FONT_FAMILY, 7)
        )
        hint.pack(side="right", padx=(0, 4))

        close_btn = tk.Label(
            title_bar, text="✕",
            bg=BG_COLOR, fg="#665577",
            font=(FONT_FAMILY, 11, "bold"),
            cursor="hand2"
        )
        close_btn.pack(side="right", padx=(0, 6))
        close_btn.bind("<Button-1>", lambda e: self._on_close())
        close_btn.bind("<Enter>", lambda e: close_btn.config(fg="#ff6688"))
        close_btn.bind("<Leave>", lambda e: close_btn.config(fg="#665577"))

        for widget in (title_bar, grip_lbl, hint):
            widget.bind("<ButtonPress-1>", self._on_drag_start)
            widget.bind("<B1-Motion>",     self._on_drag_motion)

        # Divider
        tk.Frame(inner, bg=BORDER_COLOR, height=1).pack(fill="x", padx=6, pady=(4, 0))

        # ── Control bar ──
        ctrl = tk.Frame(inner, bg=BG_COLOR)
        ctrl.pack(fill="x", padx=8, pady=(5, 2))

        self._toggle_btn = tk.Label(
            ctrl, text="▶ Start OCR",
            bg=ACCENT_COLOR, fg="white",
            font=(FONT_FAMILY, 8, "bold"),
            padx=8, pady=3, cursor="hand2"
        )
        self._toggle_btn.pack(side="left")
        self._toggle_btn.bind("<Button-1>", lambda _e: self._toggle_ocr())
        self._toggle_btn.bind("<Enter>", lambda _e: self._toggle_btn.config(bg="#9980ff"))
        self._toggle_btn.bind("<Leave>", lambda _e: self._toggle_btn.config(
            bg=ACCENT_COLOR if not self._ocr_running else "#cc4455"
        ))

        region_btn = tk.Label(
            ctrl, text="⊹ Region",
            bg="#2a2a4e", fg="#aaaacc",
            font=(FONT_FAMILY, 8, "bold"),
            padx=8, pady=3, cursor="hand2"
        )
        region_btn.pack(side="left", padx=(6, 0))
        region_btn.bind("<Button-1>", lambda _e: self._select_region())
        region_btn.bind("<Enter>", lambda _e: region_btn.config(fg="white"))
        region_btn.bind("<Leave>", lambda _e: region_btn.config(fg="#aaaacc"))

        self._region_lbl = tk.Label(
            ctrl, text="Full screen",
            bg=BG_COLOR, fg="#444466",
            font=(FONT_FAMILY, 7)
        )
        self._region_lbl.pack(side="left", padx=(6, 0))

        self._status_lbl = tk.Label(
            ctrl, text="Idle  |  Install: rapidocr-onnxruntime deep-translator Pillow" if not (HAS_PIL and HAS_OCR and HAS_TRANSLATE) else "Idle",
            bg=BG_COLOR, fg="#555577",
            font=(FONT_FAMILY, 7)
        )
        self._status_lbl.pack(side="right")

        # ── Language picker ──
        lang_bar = tk.Frame(inner, bg=BG_COLOR)
        lang_bar.pack(fill="x", padx=8, pady=(0, 3))

        tk.Label(
            lang_bar, text="Input:",
            bg=BG_COLOR, fg="#555577",
            font=(FONT_FAMILY, 7)
        ).pack(side="left")

        self._lang_btns = {}
        for code, label in [("en", "EN"), ("fr", "FR"), ("es", "ES")]:
            btn = tk.Label(
                lang_bar, text=label,
                bg="#2a2a4e", fg="#aaaacc",
                font=(FONT_FAMILY, 7, "bold"),
                padx=6, pady=1, cursor="hand2"
            )
            btn.pack(side="left", padx=(4, 0))
            btn.bind("<Button-1>", lambda _e, c=code: self._set_source_lang(c))
            self._lang_btns[code] = btn

        self._set_source_lang("en")  # highlight default

        # Mode toggle — right side of the same bar
        tk.Label(
            lang_bar, text="Mode:",
            bg=BG_COLOR, fg="#555577",
            font=(FONT_FAMILY, 7)
        ).pack(side="right", padx=(0, 4))

        self._mode_btns = {}
        for mode, label in [("manual", "Manual"), ("auto", "Auto")]:
            btn = tk.Label(
                lang_bar, text=label,
                bg="#2a2a4e", fg="#aaaacc",
                font=(FONT_FAMILY, 7, "bold"),
                padx=6, pady=1, cursor="hand2"
            )
            btn.pack(side="right", padx=(0, 4))
            btn.bind("<Button-1>", lambda _e, m=mode: self._set_mode(m))
            self._mode_btns[mode] = btn

        self._set_mode("auto")  # highlight default

        # ── OCR source text (small, dimmed) ──
        tk.Label(
            inner, text=f"OCR  [{OCR_LANG}]",
            bg=BG_COLOR, fg="#444466",
            font=(FONT_FAMILY, 7)
        ).pack(anchor="w", padx=8, pady=(4, 0))

        self._ocr_text = tk.Text(
            inner,
            bg="#0d0d1e", fg="#667788",
            font=(FONT_FAMILY, 9),
            relief="flat", wrap="word", height=2,
            padx=6, pady=4, bd=0,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            state="disabled"
        )
        self._ocr_text.pack(fill="x", padx=8)
        self._ocr_text.bind("<Button-1>", lambda e: self.root.after(10, lambda: self._on_word_click(e, TRANSLATE_TO)))

        # ── Translation output (main, bright) ──
        tk.Label(
            inner, text=f"Translation  [→ {TRANSLATE_TO}]",
            bg=BG_COLOR, fg=ACCENT_COLOR,
            font=(FONT_FAMILY, 7, "bold")
        ).pack(anchor="w", padx=8, pady=(6, 0))

        self._trans_text = tk.Text(
            inner,
            bg="#0d0d1e", fg=TEXT_COLOR,
            insertbackground=ACCENT_COLOR,
            font=(FONT_FAMILY, FONT_SIZE),
            relief="flat", wrap="word", height=4,
            padx=8, pady=6, bd=0,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=ACCENT_COLOR,
            state="disabled"
        )
        self._trans_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._trans_text.bind("<Button-1>", lambda e: self.root.after(10, lambda: self._on_word_click(e, "en")))

    def _set_source_lang(self, code):
        self._source_lang = code
        self._word_cache.clear()     # old cache no longer valid for new language
        self._last_raw_text = None   # force re-translation on next cycle
        self._translator = None      # recreate with new source language
        for c, btn in self._lang_btns.items():
            if c == code:
                btn.config(bg=ACCENT_COLOR, fg="white")
            else:
                btn.config(bg="#2a2a4e", fg="#aaaacc")

    def _set_mode(self, mode):
        # Stop any running auto loop when switching modes
        if self._ocr_running and mode != self._mode:
            self._stop_ocr()
        self._mode = mode
        for m, btn in self._mode_btns.items():
            btn.config(bg=ACCENT_COLOR if m == mode else "#2a2a4e",
                       fg="white"       if m == mode else "#aaaacc")
        if mode == "manual":
            self._toggle_btn.config(text="◉ Capture", bg="#2a6644")
        else:
            self._toggle_btn.config(text="▶ Start OCR", bg=ACCENT_COLOR)

    # ── OCR loop ─────────────────────────────────────────────────────────────────
    def _toggle_ocr(self):
        if self._mode == "manual":
            if not self._ocr_running:
                self._capture_once()
        else:
            if self._ocr_running:
                self._stop_ocr()
            else:
                self._start_ocr()

    def _capture_once(self):
        """Manual mode: single capture → OCR → translate, then idle."""
        if not HAS_PIL or not HAS_OCR:
            self._set_status("Missing packages")
            return
        self._ocr_running = True
        self._toggle_btn.config(text="… Working", bg="#888888")
        t = threading.Thread(target=self._ocr_once_worker, daemon=True)
        t.start()

    def _start_ocr(self):
        if not HAS_PIL:
            self._set_status("Missing: pip install Pillow")
            return
        if not HAS_OCR:
            self._set_status("Missing: pip install rapidocr-onnxruntime")
            return

        self._ocr_running = True
        self._toggle_btn.config(text="■ Stop OCR", bg="#cc4455")
        self._set_status("Initializing RapidOCR…")

        self._ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        self._ocr_thread.start()

    def _stop_ocr(self):
        self._ocr_running = False
        self._last_img_hash = None
        self._last_raw_text = None
        self._no_change_streak = 0
        if self._mode == "manual":
            self._toggle_btn.config(text="◉ Capture", bg="#2a6644")
        else:
            self._toggle_btn.config(text="▶ Start OCR", bg=ACCENT_COLOR)
        self._set_status("Stopped")

    def _init_paddle(self):
        """Initialize RapidOCR if not already done. Returns False on failure."""
        if self._paddle is not None:
            return True
        try:
            self._paddle = RapidOCR()
            return True
        except Exception as e:
            self.root.after(0, lambda: self._set_status(f"OCR init error: {e}"))
            self._ocr_running = False
            self.root.after(0, self._stop_ocr)
            return False

    def _run_ocr_once(self):
        """
        Capture → thumbnail check → OCR → text check → translate → display.
        Returns True if the display was updated, False if skipped.
        """
        img = ImageGrab.grab(bbox=self._capture_region)

        # Skip OCR if image hasn't changed.
        # Grayscale 64×36 = 2.3 KB to hash vs 160×90 RGB = 43 KB — 19× less work.
        thumb = img.convert("L").resize((64, 36))
        img_hash = hash(thumb.tobytes())
        if img_hash == self._last_img_hash:
            self._no_change_streak += 1
            self.root.after(0, lambda: self._set_status(
                f"No change  {time.strftime('%H:%M:%S')}"
            ))
            return False
        self._no_change_streak = 0
        self._last_img_hash = img_hash

        # Downscale before OCR — fewer pixels = much faster inference
        if img.width > MAX_OCR_WIDTH:
            ratio = MAX_OCR_WIDTH / img.width
            img = img.resize((MAX_OCR_WIDTH, int(img.height * ratio)), Image.BILINEAR)

        # OCR
        result, _ = self._paddle(np.array(img))
        lines = [
            item[1] for item in (result or [])
            if item and len(item) >= 3 and float(item[2]) > 0.5
        ]
        raw_text = " ".join(lines).strip()

        if not raw_text:
            self.root.after(0, lambda: self._set_status("No text detected"))
            return False

        # Skip translation if text unchanged
        if raw_text == self._last_raw_text:
            return False
        self._last_raw_text = raw_text

        # Translate — reuse cached session; only rebuild when source lang changes
        translated = raw_text
        if HAS_TRANSLATE:
            try:
                if self._translator is None:
                    self._translator = GoogleTranslator(
                        source=self._source_lang, target=TRANSLATE_TO
                    )
                translated = self._translator.translate(raw_text) or raw_text
            except Exception as ex:
                self._translator = None  # force rebuild next cycle
                translated = f"[Translation error: {ex}]"

        rt, tr = raw_text, translated
        self.root.after(0, lambda r=rt, t=tr: self._update_display(r, t))
        return True

    def _ocr_once_worker(self):
        """Manual mode: run one capture cycle then return to idle."""
        if not self._init_paddle():
            return
        self.root.after(0, lambda: self._set_status("Capturing…"))
        try:
            self._run_ocr_once()
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: self._set_status(f"Error: {err}"))
        self._ocr_running = False
        self.root.after(0, lambda: self._toggle_btn.config(text="◉ Capture", bg="#2a6644"))

    def _ocr_worker(self):
        """Auto mode: run capture cycle in a loop until stopped."""
        if not self._init_paddle():
            return
        self.root.after(0, lambda: self._set_status("Running"))
        while self._ocr_running:
            t0 = time.time()
            try:
                self._run_ocr_once()
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda e=err: self._set_status(f"Error: {e}"))

            # Adaptive back-off: add 0.5s per consecutive idle cycle, cap at 10s.
            # Resets to OCR_INTERVAL immediately when the screen changes.
            idle_bonus = min(self._no_change_streak * 0.5, 10.0 - OCR_INTERVAL)
            time.sleep(max(0.0, OCR_INTERVAL + idle_bonus - (time.time() - t0)))

    def _get_dpi_scale(self):
        """
        Compute the ratio between physical screenshot pixels and tkinter logical pixels.
        On a 3840x2160 screen at 200% Windows scaling, tkinter reports 1920x1080
        but ImageGrab returns 3840x2160, so the scale is 2.0.
        Result is cached after the first call.
        """
        if self._dpi_scale is None:
            img = ImageGrab.grab()
            self._dpi_scale = img.width / self.root.winfo_screenwidth()
        return self._dpi_scale

    def _select_region(self):
        """Open a fullscreen overlay for the user to drag-select a capture region."""
        # Compute DPI scale before hiding the window (screenshot needs the screen visible)
        self._get_dpi_scale()
        self.root.withdraw()

        sel = tk.Toplevel()
        sel.attributes("-fullscreen", True)
        sel.attributes("-topmost", True)
        sel.attributes("-alpha", 0.35)
        sel.configure(bg="black")
        sel.overrideredirect(True)

        canvas = tk.Canvas(sel, cursor="cross", bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        sw = sel.winfo_screenwidth()
        canvas.create_text(
            sw // 2, 40,
            text="Drag to select capture region   |   ESC to cancel   |   Right-click to reset to full screen",
            fill="white", font=(FONT_FAMILY, 12)
        )

        start = [0, 0]
        rect_id = [None]

        def on_press(e):
            start[0], start[1] = e.x, e.y
            if rect_id[0]:
                canvas.delete(rect_id[0])

        def on_drag(e):
            if rect_id[0]:
                canvas.delete(rect_id[0])
            rect_id[0] = canvas.create_rectangle(
                start[0], start[1], e.x, e.y,
                outline=ACCENT_COLOR, width=2,
                fill=ACCENT_COLOR, stipple="gray25"
            )

        def on_release(e):
            x1, y1 = min(start[0], e.x), min(start[1], e.y)
            x2, y2 = max(start[0], e.x), max(start[1], e.y)
            if x2 - x1 > 20 and y2 - y1 > 20:
                s = self._dpi_scale  # physical px per logical px (e.g. 2.0 on 4K/200%)
                self._capture_region = (
                    int(x1 * s), int(y1 * s),
                    int(x2 * s), int(y2 * s),
                )
                self._region_lbl.config(
                    text=f"{x1},{y1} → {x2},{y2}  (×{s:.1f})", fg=ACCENT_COLOR
                )
            _close()

        def on_reset(__):  # noqa: unused event arg
            self._capture_region = None
            self._region_lbl.config(text="Full screen", fg="#444466")
            _close()

        def _close():
            sel.destroy()
            self.root.deiconify()
            self.root.attributes("-topmost", True)

        canvas.bind("<ButtonPress-1>",   on_press)
        canvas.bind("<B1-Motion>",       on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        canvas.bind("<ButtonPress-3>",   on_reset)
        sel.bind("<Escape>",             lambda _e: _close())

    # ── Word panel ───────────────────────────────────────────────────────────────
    def _build_word_panel(self):
        panel = tk.Toplevel(self.root)
        panel.overrideredirect(True)
        panel.attributes("-topmost", True)
        panel.attributes("-alpha", ALPHA)
        panel.configure(bg=BG_COLOR)
        self._word_panel = panel

        # Place it to the right of the main window
        x = self.root.winfo_x() + WIN_WIDTH + 6
        y = self.root.winfo_y()
        panel.geometry(f"220x{WIN_HEIGHT}+{x}+{y}")

        outer = tk.Frame(panel, bg=BORDER_COLOR, padx=1, pady=1)
        outer.pack(fill="both", expand=True)
        inner = tk.Frame(outer, bg=BG_COLOR)
        inner.pack(fill="both", expand=True)

        # Title bar
        title_bar = tk.Frame(inner, bg=BG_COLOR, height=28)
        title_bar.pack(fill="x", padx=4, pady=(4, 0))
        title_bar.pack_propagate(False)

        title_lbl = tk.Label(
            title_bar, text="⠿ WORD LOOKUP",
            bg=BG_COLOR, fg=ACCENT_COLOR,
            font=(FONT_FAMILY, 8, "bold")
        )
        title_lbl.pack(side="left", padx=2)

        # Independent drag for the word panel
        dx, dy = [0], [0]
        def on_press(e):
            dx[0] = e.x_root - panel.winfo_x()
            dy[0] = e.y_root - panel.winfo_y()
        def on_motion(e):
            panel.geometry(f"+{e.x_root - dx[0]}+{e.y_root - dy[0]}")
        for w in (title_bar, title_lbl):
            w.bind("<ButtonPress-1>", on_press)
            w.bind("<B1-Motion>",     on_motion)

        tk.Frame(inner, bg=BORDER_COLOR, height=1).pack(fill="x", padx=4, pady=(2, 0))

        # Scrollable list using a Text widget
        container = tk.Frame(inner, bg=BG_COLOR)
        container.pack(fill="both", expand=True, padx=4, pady=4)

        scrollbar = tk.Scrollbar(container, width=6, relief="flat",
                                 bg=BG_COLOR, troughcolor="#0d0d1e")
        scrollbar.pack(side="right", fill="y")

        self._word_text = tk.Text(
            container,
            bg="#0d0d1e", fg=TEXT_COLOR,
            font=(FONT_FAMILY, 10),
            relief="flat", wrap="none",
            padx=8, pady=4, bd=0,
            highlightthickness=0,
            state="disabled",
            cursor="arrow",
            spacing1=2, spacing3=4,
            yscrollcommand=scrollbar.set,
        )
        self._word_text.pack(fill="both", expand=True)
        scrollbar.config(command=self._word_text.yview)

        # Text tags for styling
        self._word_text.tag_configure("src",     foreground="#aaaacc",  font=(FONT_FAMILY, 9))
        self._word_text.tag_configure("tgt",     foreground=ACCENT_COLOR, font=(FONT_FAMILY, 12, "bold"))
        self._word_text.tag_configure("pending", foreground="#444466",  font=(FONT_FAMILY, 10, "italic"))
        self._word_text.tag_configure("sep",     foreground=BORDER_COLOR, font=(FONT_FAMILY, 7))

    def _refresh_word_panel(self, words, gen=None):
        """Redraw word panel rows using current cache (main thread only)."""
        if gen is not None and gen != self._word_gen:
            return   # stale callback, discard
        wt = self._word_text
        wt.config(state="normal")
        wt.delete("1.0", "end")
        for i, word in enumerate(words):
            if i:
                wt.insert("end", "─" * 22 + "\n", "sep")
            wt.insert("end", word + "\n", "src")
            trans = self._word_cache.get(word)
            wt.insert("end", (trans if trans else "…") + "\n",
                      "tgt" if trans else "pending")
        wt.config(state="disabled")

    def _schedule_word_update(self, raw_text):
        """Extract unique words, clear the panel, show cached hits, queue new ones."""
        words = list(dict.fromkeys(
            w.lower() for w in re.sub(r"[^a-zA-Z]", " ", raw_text).split()
            if len(w) > 1
        ))

        # Bump generation so any running thread knows it is stale
        self._word_gen += 1
        gen = self._word_gen

        # Show current state immediately (cached words filled in, rest show "…")
        self._refresh_word_panel(words, gen)

        new_words = [w for w in words if w not in self._word_cache]
        if new_words:
            threading.Thread(
                target=self._translate_words_bg,
                args=(gen, new_words, words),
                daemon=True
            ).start()

    def _translate_words_bg(self, gen, new_words, all_words):
        """Translate all new words in one batched HTTP call, then refresh once."""
        if gen != self._word_gen:
            return
        try:
            # Join with newlines so Google Translate keeps them as separate lines
            batch = "\n".join(new_words)
            result = GoogleTranslator(
                source=self._source_lang, target=TRANSLATE_TO
            ).translate(batch) or ""
            translations = result.split("\n")
            for i, word in enumerate(new_words):
                self._word_cache[word] = translations[i].strip() if i < len(translations) else "—"
        except Exception:
            for word in new_words:
                self._word_cache[word] = "—"
        if gen == self._word_gen:
            snapshot = list(all_words)
            self.root.after(0, lambda s=snapshot, g=gen: (
                self._refresh_word_panel(s, g) if g == self._word_gen else None
            ))

    def _update_display(self, raw_text, translated):
        self._set_text(self._ocr_text, raw_text)
        self._set_text(self._trans_text, translated)
        self._set_status(f"Updated {time.strftime('%H:%M:%S')}")
        self._schedule_word_update(raw_text)

    # ── Word popup ───────────────────────────────────────────────────────────────
    def _on_word_click(self, event, target_lang):
        widget = event.widget
        try:
            idx = widget.index(f"@{event.x},{event.y}")
            word_start = widget.index(f"{idx} wordstart")
            word_end   = widget.index(f"{idx} wordend")
            word = widget.get(word_start, word_end).strip()
        except Exception:
            return
        if word and any(c.isalpha() or '\u4e00' <= c <= '\u9fff' for c in word):
            self._show_word_popup(word, event.x_root, event.y_root, target_lang)

    def _show_word_popup(self, word, x, y, target_lang):
        # Close any existing word popup
        if hasattr(self, '_word_popup') and self._word_popup:
            try:
                self._word_popup.destroy()
            except Exception:
                pass

        popup = tk.Toplevel(self.root)
        popup.overrideredirect(True)
        popup.attributes("-topmost", True)
        popup.attributes("-alpha", 0.95)
        popup.configure(bg=BG_COLOR)
        self._word_popup = popup

        # Position near the click, nudge inward if near screen edge
        sw = self.root.winfo_screenwidth()
        px = min(x + 15, sw - 220)
        py = y + 15
        popup.geometry(f"+{px}+{py}")

        outer = tk.Frame(popup, bg=ACCENT_COLOR, padx=1, pady=1)
        outer.pack()
        inner = tk.Frame(outer, bg=BG_COLOR)
        inner.pack()

        # Word being looked up
        tk.Label(
            inner, text=word,
            bg=BG_COLOR, fg="white",
            font=(FONT_FAMILY, 13, "bold"),
            padx=14, pady=(10, 2)
        ).pack()

        divider = tk.Frame(inner, bg=BORDER_COLOR, height=1)
        divider.pack(fill="x", padx=10)

        # Translation result (starts as "…", filled in by thread)
        result_lbl = tk.Label(
            inner, text="…",
            bg=BG_COLOR, fg=ACCENT_COLOR,
            font=(FONT_FAMILY, 15),
            padx=14, pady=(6, 4)
        )
        result_lbl.pack()

        hint = tk.Label(
            inner, text="click to close",
            bg=BG_COLOR, fg="#444466",
            font=(FONT_FAMILY, 6),
            padx=14, pady=(0, 8)
        )
        hint.pack()

        def close_popup():
            try:
                popup.destroy()
            except Exception:
                pass

        popup.bind("<Button-1>", lambda *_: close_popup())
        popup.after(6000, close_popup)

        # Translate in background so UI stays responsive
        def do_translate():
            try:
                result = GoogleTranslator(source="auto", target=target_lang).translate(word)
                text = result or word
            except Exception:
                text = "—"
            if popup.winfo_exists():
                popup.after(0, lambda: result_lbl.config(text=text))

        threading.Thread(target=do_translate, daemon=True).start()

    def _set_text(self, widget, text):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.config(state="disabled")

    def _set_status(self, msg):
        self._status_lbl.config(text=msg)

    # ── Drag logic ───────────────────────────────────────────────────────────────
    def _bind_drag(self):
        self.root.bind("<ButtonPress-1>", self._on_drag_start)
        self.root.bind("<B1-Motion>",     self._on_drag_motion)

    def _on_drag_start(self, event):
        self._drag_x = event.x_root - self.root.winfo_x()
        self._drag_y = event.y_root - self.root.winfo_y()

    def _on_drag_motion(self, event):
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    # ── Hotkey ───────────────────────────────────────────────────────────────────
    def _register_hotkey(self):
        if not HAS_KEYBOARD:
            print(
                "⚠  'keyboard' package not found.\n"
                "   Install it with:  pip install keyboard\n"
            )
            return
        keyboard.add_hotkey(HOTKEY, lambda: self.root.after(0, self._toggle_visibility))

    def _toggle_visibility(self):
        if self.visible:
            self.root.withdraw()
            if self._word_panel:
                self._word_panel.withdraw()
            self.visible = False
        else:
            self.root.deiconify()
            self.root.attributes("-topmost", True)
            if self._word_panel:
                self._word_panel.deiconify()
                self._word_panel.attributes("-topmost", True)
            self.visible = True

    def _on_close(self):
        self._ocr_running = False
        self.root.destroy()

    # ── Run ──────────────────────────────────────────────────────────────────────
    def run(self):
        print("✅ Translation Overlay is running.")
        print(f"   Hotkey : {HOTKEY.upper()} → toggle show/hide")
        print(f"   OCR    : {OCR_LANG} → {TRANSLATE_TO} every {OCR_INTERVAL}s")
        print("   Close  : click the ✕ button on the overlay\n")
        if not HAS_PIL:
            print("⚠  pip install Pillow")
        if not HAS_OCR:
            print("⚠  pip install rapidocr-onnxruntime")
        if not HAS_TRANSLATE:
            print("⚠  pip install deep-translator")
        self.root.mainloop()
        if HAS_KEYBOARD:
            keyboard.unhook_all()


# ── Entry point ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = TranslationOverlay()
    app.run()
