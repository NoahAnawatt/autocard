#!/usr/bin/env python3
import sys, subprocess, curses, time, json, random, re
from pathlib import Path
from datetime import datetime
import threading, queue

# =====================
# DEFAULT CONFIG + DESCRIPTIONS
# =====================
DEFAULT_CONFIG = {
    "model": "llama3:8b",
    "chunk_size": 1600,
    "overlap": 200,
    "num_passes": 2,
    "topic": "Web Programming",
    "max_tokens": 2048,
    "chunk_order": "sequential",
    "output_format": "tsv",
    "num_threads": 2,
    "target_cards": 10
}

CONFIG_DESC = {
    "model": "Ollama model to use",
    "chunk_size": "Number of characters per chunk",
    "overlap": "Number of overlapping characters between chunks",
    "num_passes": "Number of passes per chunk",
    "topic": "Topic for flashcards",
    "max_tokens": "Maximum tokens per Ollama request",
    "chunk_order": "Order of chunks: sequential, reverse, random",
    "output_format": "Output file format: tsv, csv, jsonl",
    "num_threads": "Number of concurrent Ollama workers" ,
    "target_cards": "Number of cards to make per chunk"
}

CONFIG_FILE = Path("flashcards.conf")

# =====================
# CONFIG UTILITIES
# =====================
def load_config():
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            for k, v in DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
        except Exception:
            return DEFAULT_CONFIG.copy()
    else:
        return DEFAULT_CONFIG.copy()

def save_config(cfg):
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Error saving config: {e}")

# =====================
# CHUNKING
# =====================
def chunk_text(text, size, overlap):
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        yield text[start:end]
        start = end - overlap if end - overlap > start else end

# =====================
# PROMPT BUILDER
# =====================
PROMPT_TEMPLATE = """
You are generating HIGH-DENSITY and EXHAUSTIVE flashcards.
Ignore any content that is not relevant to the topic of {topic}.
Extract EVERY detail. Every fact → at least one flashcard.
Each flashcard should be useful for studying for an exam.
If there are no relevant details, print "NONE".
Aim to produce {limit} cards, unless there are more important details.
Output EXACTLY one question<TAB>answer per line, no numbering, no markdown.

Text:
=========================
{chunk}
=========================

Flashcards:
"""

if Path("./prompt.acd").is_file():
    with open("./prompt.acd") as promptFile:
        PROMPT_TEMPLATE = promptFile.read()
        print('Loaded alternate prompt')

def build_prompt(topic, chunk, limit):
    return PROMPT_TEMPLATE.format(topic=topic, chunk=chunk,limit=limit)

# =====================
# THREAD WORKER FOR OLAMA
# =====================
def ollama_worker(model, prompt, line_queue, stop_event):
    """Runs Ollama subprocess and pushes lines to queue"""
    try:
        proc = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        proc.stdin.write(prompt)
        proc.stdin.close()
        for line in proc.stdout:
            if stop_event.is_set():
                break
            line_queue.put(line.rstrip("\n"))
        proc.stdout.close()
        proc.wait()
    except Exception as e:
        line_queue.put(f"[ERROR] Ollama worker failed: {e}")

def chunk_worker(cfg, task_queue, line_queue, stop_event):
    """Worker that processes DIFFERENT chunks by pulling from task_queue."""
    while not stop_event.is_set():
        try:
            chunk_index, pass_num, chunk = task_queue.get(timeout=0.1)
        except queue.Empty:
            return

        prompt = build_prompt(cfg["topic"], chunk, cfg["target_cards"])

        try:
            proc = subprocess.Popen(
                ["ollama", "run", cfg["model"]],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            proc.stdin.write(prompt)
            proc.stdin.close()

            for line in proc.stdout:
                if stop_event.is_set():
                    break
                line_queue.put((chunk_index, pass_num, line.rstrip("\n")))

            proc.stdout.close()
            proc.wait()

        except Exception as e:
            line_queue.put((chunk_index, pass_num, f"[ERROR] Worker failed: {e}"))

        task_queue.task_done()

# =====================
# FLASHCARD EXTRACTION (TAB OR <TAB>)
# =====================
def extract_lines(lines, output_format="tsv"):
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "<TAB>" in line:
            line = line.replace("<TAB>", "\t")

        if "\t" in line:
            q, a = map(str.strip, line.split("\t", 1))
        elif "  " in line:
            parts = re.split(r"\s{2,}", line, maxsplit=1)
            if len(parts) != 2:
                continue
            q, a = map(str.strip, parts)
        else:
            continue

        if not q or not a:
            continue

        if output_format == "tsv":
            yield f"{q}\t{a}\n"
        elif output_format == "csv":
            import csv
            from io import StringIO
            out = StringIO()
            writer = csv.writer(out)
            writer.writerow([q, a])
            yield out.getvalue()
        elif output_format == "jsonl":
            yield json.dumps({"question": q, "answer": a}) + "\n"

# =====================
# DASHBOARD
# =====================
class Dashboard:
    SPINNER = ["|","/","-","\\"]

    def __init__(self, stdscr, total_chunks):
        self.stdscr = stdscr
        self.total_chunks = total_chunks
        self.chunk = 0
        self.pass_num = 0
        self.flashcards = 0
        self.logs = []
        self.spinner_index = 0
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN,-1)
        curses.init_pair(2, curses.COLOR_YELLOW,-1)
        curses.init_pair(3, curses.COLOR_RED,-1)
        curses.init_pair(4, curses.COLOR_CYAN,-1)
        curses.init_pair(5, curses.COLOR_WHITE,-1)

    def log_event(self, msg, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{ts}][{level}] {msg}")
        if len(self.logs) > 20:
            self.logs = self.logs[-20:]

    def draw(self):
        self.stdscr.erase()
        h,w = self.stdscr.getmaxyx()
        try:
            self.stdscr.addstr(0,0,"FLASHCARD GENERATOR — Color Dashboard",curses.color_pair(1)|curses.A_BOLD)
            self.stdscr.addstr(2,0,f"Chunk: {self.chunk}/{self.total_chunks}",curses.color_pair(4))
            self.stdscr.addstr(3,0,f"Pass:  {self.pass_num}",curses.color_pair(4))
            pct = int((self.chunk/self.total_chunks)*100) if self.total_chunks else 0
            bar_width = min(30, w-20)
            filled = int(bar_width*pct/100)
            bar = "["+"#"*filled+"-"*(bar_width-filled)+"]"
            self.stdscr.addstr(4,0,f"Progress: {bar} {pct}%",curses.color_pair(1))
            spinner = self.SPINNER[self.spinner_index % len(self.SPINNER)]
            self.stdscr.addstr(5,0,f"Running: {spinner}",curses.color_pair(2))
            self.stdscr.addstr(6,0,f"Flashcards extracted: {self.flashcards}",curses.color_pair(1))
            self.stdscr.addstr(8,0,"Logs:",curses.color_pair(5)|curses.A_BOLD)
            for i, log in enumerate(self.logs[-12:]):
                if 9+i < h-2:
                    self.stdscr.addstr(9+i,2,log[:w-4])
            self.stdscr.addstr(h-2,0,"Press 'c' for config | 'q' to quit",curses.color_pair(5)|curses.A_BOLD)
        except curses.error:
            pass
        self.stdscr.refresh()
        self.spinner_index+=1

# =====================
# CONFIG MODAL
# =====================
def config_modal(stdscr, cfg, dashboard=None):
    curses.curs_set(1)
    h,w = stdscr.getmaxyx()
    win_h = len(cfg)+4
    win = curses.newwin(win_h, w-4, max(0,h//2-win_h//2),2)
    win.keypad(True)
    win.nodelay(True)
    win.box()
    win.addstr(0,2,"CONFIGURATION MENU (ESC to exit)")

    keys = list(cfg.keys())
    selected = 0
    input_mode = False
    buffer = ""
    while True:
        if dashboard: dashboard.draw()
        for i,k in enumerate(keys):
            val=str(cfg[k])
            desc = CONFIG_DESC.get(k,"")
            prefix = "-> " if i==selected else "   "
            try: win.addstr(1+i,2,f"{prefix}{k}: {val} ({desc})".ljust(w-6))
            except curses.error: pass
        win.refresh()
        c = win.getch()
        if c==-1: time.sleep(0.05); continue
        if input_mode:
            if c in [10,13]:
                if buffer.isdigit(): cfg[keys[selected]]=int(buffer)
                else: cfg[keys[selected]]=buffer
                buffer=""; input_mode=False
            elif c==27: buffer=""; input_mode=False
            elif 32<=c<=126: buffer+=chr(c)
        else:
            if c==27: break
            elif c in [curses.KEY_UP, ord('k')]: selected=max(0,selected-1)
            elif c in [curses.KEY_DOWN, ord('j')]: selected=min(len(keys)-1,selected+1)
            elif c in [10,13]: input_mode=True; buffer=""
    del win
    curses.curs_set(0)
    save_config(cfg)
    if dashboard: dashboard.draw()

# =====================
# GENERATOR LOOP
# =====================
def run_generator(stdscr, infile, outfile, cfg):
    curses.curs_set(0)

    # Load input text & chunk it
    text = Path(infile).read_text(encoding="utf-8")
    chunks = list(chunk_text(text, cfg["chunk_size"], cfg["overlap"]))
    if cfg["chunk_order"] == "reverse":
        chunks = list(reversed(chunks))
    elif cfg["chunk_order"] == "random":
        random.shuffle(chunks)

    dash = Dashboard(stdscr, len(chunks))
    dash.draw()

    # Reset output file
    Path(outfile).write_text("", encoding="utf-8")
    tsv_count = 0

    # Shared thread state
    stop_event = threading.Event()
    line_queue = queue.Queue()
    task_queue = queue.Queue()
    write_lock = threading.Lock()

    # -------------------------
    # Build task queue
    # -------------------------
    for ci, chunk in enumerate(chunks, start=1):
        chunk_preview = chunk.replace("\n", " ").strip()
        if len(chunk_preview) > 100:
            chunk_preview = chunk_preview[:100] + "..."
        dash.log_event(f"Chunk preview: {chunk_preview}", "DEBUG")

        for p in range(1, cfg["num_passes"] + 1):
            task_queue.put((ci, p, chunk))

    # -------------------------
    # Start parallel workers
    # -------------------------
    workers = []
    for _ in range(cfg["num_threads"]):
        t = threading.Thread(
            target=chunk_worker,
            args=(cfg, task_queue, line_queue, stop_event),
            daemon=True
        )
        t.start()
        workers.append(t)

    dash.log_event(f"Started {len(workers)} parallel chunk workers", "INFO")

    # -------------------------
    # Process model output lines
    # -------------------------
    while any(w.is_alive() for w in workers) or not line_queue.empty():
        try:
            ci, p, line = line_queue.get(timeout=0.1)
        except queue.Empty:
            dash.draw()
            continue

        dash.chunk = ci
        dash.pass_num = p
        dash.log_event(f"Ollama output: {line}", "INFO")

        # Extract flashcards from the line
        for extracted in extract_lines([line], cfg["output_format"]):
            with write_lock:
                with open(outfile, "a", encoding="utf-8") as f:
                    f.write(extracted)

            tsv_count += 1
            dash.flashcards = tsv_count
            dash.log_event(f"Extracted: {extracted.strip()}", "DEBUG")

        line_queue.task_done()
        dash.draw()

        # Handle keyboard input
        stdscr.nodelay(True)
        key = stdscr.getch()

        if key == ord('q'):
            stop_event.set()
            break

        elif key == ord('c'):
            # Pause workers
            stop_event.set()
            for w in workers:
                w.join()

            # change config
            config_modal(stdscr, cfg, dashboard=dash)

            # Resume
            stop_event.clear()
            workers = []
            for _ in range(cfg["num_threads"]):
                t = threading.Thread(
                    target=chunk_worker,
                    args=(cfg, task_queue, line_queue, stop_event),
                    daemon=True
                )
                t.start()
                workers.append(t)
            dash.log_event("Workers restarted after config change", "DEBUG")

    # -------------------------
    # Shutdown
    # -------------------------
    stop_event.set()
    task_queue.join()

    for w in workers:
        w.join()

    dash.log_event("All chunks processed. Done!", "INFO")
    dash.draw()

    stdscr.nodelay(False)
    stdscr.getch()

# =====================
# MAIN
# =====================
def main(stdscr):
    if len(sys.argv)!=3:
        print("Usage: flashcards.py [input.txt] [output.tsv]")
        sys.exit(1)
    infile = sys.argv[1]
    outfile = sys.argv[2]
    cfg = load_config()
    config_modal(stdscr,cfg)
    run_generator(stdscr,infile,outfile,cfg)

if __name__=="__main__":
    curses.wrapper(main)

