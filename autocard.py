#!/usr/bin/env python3
import sys, subprocess, curses, time, json, random, re
from pathlib import Path
from datetime import datetime

VERSION = 2.5

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
    "target_cards": 10,
    "validate": "true"
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
    "target_cards": "Number of cards per chunk",
    "validate": "Remove low-quality flashcards: true,false"
}

CONFIG_FILE = Path("flashcards.conf")
REJECTED_FILE = "rejected.tsv"

# =====================
# PROMPTS
# =====================
PROMPT_TEMPLATE = """
You are generating HIGH-DENSITY and EXHAUSTIVE flashcards.
Ignore any content that is not relevant to the topic of {topic}.
Extract EVERY detail. Every fact → at least one flashcard.
Each flashcard should be useful for studying for an exam.
Each flashcard must have a stand-alone question. 
Cards about key terms must ask the question as "What is a [key term]?"
If there are no relevant details, print "NONE".
Aim to produce {limit} cards.
Output EXACTLY one question<TAB>answer per line, no numbering, no markdown.

Text:
{chunk}

Flashcards:
"""

VALIDATION_PROMPT = """
Verify that the following flashcard is useful and relevant.
It should regard the topic(s) of {topic}.
Respond only YES if the card is good and only NO if the card is bad.
"""

if Path("./prompt.acd").is_file():
    with open("./prompt.acd") as f:
        PROMPT_TEMPLATE = f.read()
        print("Loaded alternate prompt")

def build_prompt(topic, chunk, limit):
    return PROMPT_TEMPLATE.format(topic=topic, chunk=chunk, limit=limit)

# =====================
# UTILITIES
# =====================
def check_ollama_running():
    try:
        subprocess.run(["ollama", "list"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def load_config():
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            for k,v in DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k]=v
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

def chunk_text(text, size, overlap):
    start = 0
    n = len(text)
    while start < n:
        end = min(start+size, n)
        yield text[start:end]
        start = end - overlap if end - overlap > start else end

def extract_lines(lines, output_format="tsv"):
    for line in lines:
        line = line.strip()
        if not line or line.upper() == "NONE": continue
        if "<TAB>" in line: line=line.replace("<TAB>","\t")
        if "\t" in line:
            q,a = map(str.strip,line.split("\t",1))
        elif "  " in line:
            parts = re.split(r"\s{2,}", line, maxsplit=1)
            if len(parts)!=2: continue
            q,a = map(str.strip,parts)
        else:
            continue
        if not q or not a: continue
        if output_format=="tsv": yield f"{q}\t{a}\n"
        elif output_format=="csv":
            import csv
            from io import StringIO
            out=StringIO()
            writer=csv.writer(out)
            writer.writerow([q,a])
            yield out.getvalue()
        elif output_format=="jsonl":
            yield json.dumps({"question":q,"answer":a})+"\n"

# =====================
# DASHBOARD
# =====================
class Dashboard:
    SPINNER = ["|","/","-","\\"]
    def __init__(self, stdscr, total_chunks, max_rows=10):
        self.stdscr = stdscr
        self.total_chunks = total_chunks
        self.chunk = 0
        self.pass_num = 0
        self.generated = 0
        self.validated = 0
        self.rejected = 0
        self.logs = []
        self.live_lines = []
        self.spinner_index = 0
        self.chunk_pass_history = []
        self.max_rows = max_rows
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)
        curses.init_pair(5, curses.COLOR_WHITE, -1)

    def log_event(self, msg, level="INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{ts}][{level}] {msg}"
        self.logs.append(formatted)

    def draw_graph(self, row, w):
        total = max(1, self.generated)
        bar_width = w - 40
        val_len = int(self.validated / total * bar_width)
        rej_len = int(self.rejected / total * bar_width)
        pending_len = max(0, bar_width - val_len - rej_len)
        x = 0
        if val_len: self.stdscr.addstr(row, x, "#"*val_len, curses.color_pair(1)); x+=val_len
        if rej_len: self.stdscr.addstr(row, x, "!"*rej_len, curses.color_pair(3)); x+=rej_len
        if pending_len: self.stdscr.addstr(row, x, "."*pending_len, curses.color_pair(4))
        label = f"Generated: {self.generated} Validated: {self.validated} Rejected: {self.rejected}"
        self.stdscr.addstr(row, max(x+1, w-len(label)-1), label, curses.color_pair(5)|curses.A_BOLD)

    def draw_chunk_matrix(self, start_row, start_col, h, w):
        history = self.chunk_pass_history[-h:]
        for i, chunk_row in enumerate(history):
            y = start_row + i
            x = start_col
            for gen,val,rej in chunk_row:
                total = max(1, gen)
                val_len = int(val / total * 1)
                rej_len = int(rej / total * 1)
                if val_len: self.stdscr.addstr(y,x,"#",curses.color_pair(1))
                elif rej_len: self.stdscr.addstr(y,x,"!",curses.color_pair(3))
                else: self.stdscr.addstr(y,x,".",curses.color_pair(4))
                x += 1

    def draw(self):
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()
        try:
            self.stdscr.addstr(0,0,f"Autocard {VERSION}",curses.color_pair(1)|curses.A_BOLD)
            self.stdscr.addstr(2,0,f"Chunk: {self.chunk}/{self.total_chunks}",curses.color_pair(4))
            self.stdscr.addstr(3,0,f"Pass:  {self.pass_num}",curses.color_pair(4))
            pct = int((self.chunk/self.total_chunks)*100) if self.total_chunks else 0
            bar_width = min(30,w-20)
            filled = int(bar_width*pct/100)
            bar = "[" + "#"*filled + "-"*(bar_width-filled) + "]"
            self.stdscr.addstr(4,0,f"Progress: {bar} {pct}%",curses.color_pair(1))
            spinner = self.SPINNER[self.spinner_index % len(self.SPINNER)]
            self.stdscr.addstr(5,0,f"Running: {spinner}",curses.color_pair(2))

            # Chunk/pass matrix
            graph_w = 30
            graph_h = min(self.max_rows, len(self.chunk_pass_history))
            self.draw_chunk_matrix(2, w - graph_w - 1, graph_h, graph_w)

            # Main graph
            graph_row = 7
            self.draw_graph(graph_row, w)

            # Live generated lines
            live_start = graph_row + 2
            max_lines = h - live_start - 2
            lines_to_show = self.live_lines[-max_lines:] if self.live_lines else []
            for i in range(max_lines):
                if i < len(lines_to_show):
                    line = lines_to_show[i]
                    self.stdscr.addstr(live_start + i, 0, line[:w-1], curses.color_pair(5))
                else:
                    self.stdscr.addstr(live_start + i, 0, " "*(w-1))

            self.stdscr.addstr(h-1,0,"Press 'c' for config | 'q' to quit",curses.color_pair(5)|curses.A_BOLD)
        except curses.error:
            pass
        self.stdscr.refresh()
        self.spinner_index += 1

# =====================
# CONFIG MODAL
# =====================
def config_modal(stdscr,cfg,dashboard=None):
    curses.curs_set(1)
    h,w=stdscr.getmaxyx()
    win_h=len(cfg)+4
    win=curses.newwin(win_h,w-4,max(0,h//2-win_h//2),2)
    win.keypad(True)
    win.nodelay(True)
    win.box()
    win.addstr(0,2,"CONFIGURATION MENU (ESC to exit)")
    keys=list(cfg.keys())
    selected=0
    input_mode=False
    buffer=""
    while True:
        if dashboard: dashboard.draw()
        for i,k in enumerate(keys):
            val=str(cfg[k])
            desc=CONFIG_DESC.get(k,"")
            prefix="-> " if i==selected else "   "
            try: win.addstr(1+i,2,f"{prefix}{k}: {val} ({desc})".ljust(w-6))
            except curses.error: pass
        win.refresh()
        c=win.getch()
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
# GENERATION + VALIDATION + LIVE DISPLAY
# =====================
def run_generator(stdscr,infile,outfile,cfg):
    curses.curs_set(0)
    text = Path(infile).read_text(encoding="utf-8")
    chunks = list(chunk_text(text, cfg["chunk_size"], cfg["overlap"]))
    if cfg["chunk_order"]=="reverse": chunks=list(reversed(chunks))
    elif cfg["chunk_order"]=="random": random.shuffle(chunks)

    dash = Dashboard(stdscr,len(chunks))
    dash.draw()
    dash.log_event("Starting up ...", "INFO")

    Path(outfile).write_text("",encoding="utf-8")
    Path(REJECTED_FILE).write_text("",encoding="utf-8")

    for ci, chunk in enumerate(chunks, start=1):
        dash.chunk = ci
        dash.pass_num = 0
        dash.live_lines = []
        chunk_generated = []

        for p in range(1, cfg["num_passes"]+1):
            dash.pass_num = p
            prompt = build_prompt(cfg["topic"], chunk, cfg["target_cards"])
            try:
                proc = subprocess.Popen(
                    ["ollama","run",cfg["model"]],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True, bufsize=1
                )
                proc.stdin.write(prompt+"\n")
                proc.stdin.flush()
                proc.stdin.close()

                while True:
                    line = proc.stdout.readline()
                    if not line:
                        if proc.poll() is not None: break
                        time.sleep(0.01)
                        dash.draw()
                        stdscr.nodelay(True)
                        key = stdscr.getch()
                        if key==ord('q'): return
                        elif key==ord('c'): config_modal(stdscr,cfg,dashboard=dash)
                        continue
                    line = line.strip()
                    if not line: continue
                    chunk_generated.append(line)
                    dash.generated += 1
                    dash.live_lines.append(line)
                    dash.log_event(f"Generated card: {line}", "INFO")
                    dash.draw()
            except Exception as e:
                dash.log_event(f"[ERROR] Generation failed: {e}", "ERROR")

        # Validation
        validated_lines = []
        for line in chunk_generated:
            if cfg.get("validate","true").lower() != "true":
                validated_lines.append(line)
                dash.validated += 1
                continue
            val_prompt = VALIDATION_PROMPT.format(topic=cfg["topic"]) + "\n" + line
            try:
                val_proc = subprocess.Popen(
                    ["ollama","run",cfg["model"]],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout_val,_ = val_proc.communicate(val_prompt, timeout=30)
                if "NO" in stdout_val.upper():
                    dash.rejected += 1
                    with open(REJECTED_FILE,"a",encoding="utf-8") as f:
                        f.write(line+"\n")
                    dash.log_event(f"Rejected card: {line}", "WARN")
                else:
                    validated_lines.append(line)
                    dash.validated += 1
                    dash.log_event(f"Validated card: {line}", "INFO")
                dash.draw()
            except Exception as e:
                dash.log_event(f"[ERROR] Validation failed: {e}", "ERROR")
                dash.draw()

        with open(outfile,"a",encoding="utf-8") as f:
            for line in validated_lines:
                f.write(line+"\n")

        dash.chunk_pass_history.append([(len(chunk_generated), dash.validated, dash.rejected)])

    dash.log_event("All chunks processed. Done!","INFO")
    dash.draw()
    stdscr.nodelay(False)
    stdscr.getch()

# =====================
# MAIN
# =====================
def main(stdscr):
    infile = sys.argv[1]
    outfile = sys.argv[2]
    cfg = load_config()
    config_modal(stdscr,cfg)
    run_generator(stdscr,infile,outfile,cfg)

def check_environment():
    if len(sys.argv)!=3:
        print("Usage: flashcards.py [input.txt] [output.tsv]",flush=True)
        sys.exit(1)
    if not check_ollama_running():
        print("ERROR: Ollama is not running. Start it with 'ollama serve'",flush=True)
        sys.exit(2)

if __name__=="__main__":
    check_environment()
    curses.wrapper(main)

