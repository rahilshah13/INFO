import requests, mwparserfromhell, os, random, re, threading, json, shutil, traceback, time, subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from concurrent.futures import ThreadPoolExecutor

HEADERS = {'User-Agent': 'WikiDataExtractor/1.0 (admin@example.com)'}

# Point everything directly to the mounted Docker volume container path
OUTPUT_DIR = "/info_txt_volume/"
STATE_FILE = os.path.join(OUTPUT_DIR, "downloaded_ids.txt")
PARSED_STATE_FILE = os.path.join(OUTPUT_DIR, "parsed_ids.txt")
WORDS_PL_PATH = os.path.join(OUTPUT_DIR, "words.pl")
PREDICATES_PL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predicates.pl")
if not os.path.exists(PREDICATES_PL_PATH):
    PREDICATES_PL_PATH = "/INFO_SRC/predicates.pl"
SIZE_LIMIT_GB = 2

print(f"[INIT] Ensuring volume directory exists at: {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(WORDS_PL_PATH) and os.path.exists("words.pl"):
    print(f"[INIT] Seeding words.pl into volume from local path...")
    shutil.copy("words.pl", WORDS_PL_PATH)
elif os.path.exists(WORDS_PL_PATH):
    print(f"[INIT] Found existing words.pl in volume.")
else:
    print(f"[INIT] Warning: words.pl not found locally or in volume.")

word_locations = {}
sentence_formats = {}      # Track hierarchical sentence formats
flat_pos_formats = {}      # Track flat parts of speech sequences
sentence_lengths = []      # Track sentence lengths in words
location_lock = threading.Lock()
format_lock = threading.Lock()
metrics_lock = threading.Lock()
prolog_lock = threading.Lock()  # Protect interactive Prolog pipe
words_db_set = set()
is_processing = False
JOB_STATE = {"state": "IDLE", "progress": 0, "total": 0, "pending_articles": 0, "current_sentence": "", "error": None}

# Persistent long-lived Trealla Prolog Interactive Worker Process
prolog_proc = None

def init_prolog_worker():
    global prolog_proc
    if prolog_proc is not None:
        try:
            prolog_proc.poll()
            if prolog_proc.returncode is None:
                return
        except Exception:
            pass

    if os.path.exists(PREDICATES_PL_PATH) and os.path.exists(WORDS_PL_PATH):
        cmd = ["tpl", PREDICATES_PL_PATH, WORDS_PL_PATH]
    elif os.path.exists(WORDS_PL_PATH):
        cmd = ["tpl", WORDS_PL_PATH]
    else:
        cmd = ["tpl"]

    try:
        print("[PROLOG] Starting long-lived interactive Trealla Prolog daemon process...")
        prolog_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    except Exception as e:
        print(f"[PROLOG-ERROR] Failed to start persistent tpl process: {e}")
        prolog_proc = None

def update_job_state(state, progress, total, current_sentence="", error=None):
    global JOB_STATE
    downloaded = 0
    parsed = 0
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                downloaded = sum(1 for _ in f)
        if os.path.exists(PARSED_STATE_FILE):
            with open(PARSED_STATE_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                parsed = sum(1 for _ in f)
    except Exception:
        pass
    pending = max(0, downloaded - parsed)
    JOB_STATE = {
        "state": state,
        "progress": progress,
        "total": total,
        "pending_articles": pending,
        "current_sentence": current_sentence,
        "error": error
    }

def evaluate_sentence_via_prolog(sentence_str):
    global prolog_proc
    words = re.findall(r'\b[a-zA-Z0-9]+\b', sentence_str.lower())
    if len(words) < 2:
        return None, None
    
    prolog_list = "[" + ", ".join(words) + "]"
    query_hierarchical = f"english:sentence_types({prolog_list}, T), write(T), nl."
    query_flat = f"english:words_to_parts_of_speech({prolog_list}, F), write(F), nl."

    with prolog_lock:
        init_prolog_worker()
        if not prolog_proc or prolog_proc.poll() is not None:
            return None, None
        
        try:
            # Set a non-blocking timeout strategy via process check / readline wrapper if needed, 
            # but with our bounded Prolog grammar, queries return instantly.
            prolog_proc.stdin.write(query_hierarchical + "\n")
            prolog_proc.stdin.flush()
            hier_res = prolog_proc.stdout.readline().strip()

            prolog_proc.stdin.write(query_flat + "\n")
            prolog_proc.stdin.flush()
            flat_res = prolog_proc.stdout.readline().strip()

            def clean_res(res):
                if not res or res == "false." or res.startswith("error"):
                    return None
                if "=" in res:
                    parts = res.split("=", 1)
                    res = parts[1].strip().rstrip(".")
                return res

            return clean_res(hier_res), clean_res(flat_res)
        except Exception as e:
            print(f"[PROLOG-PIPE-ERROR] {e}")
            prolog_proc = None
            
    return None, None

def load_words_pl_vocabulary():
    global words_db_set
    print("[VOCAB] Loading Trealla Prolog words.pl vocabulary set...")
    if not os.path.exists(WORDS_PL_PATH):
        print("[VOCAB] words.pl path does not exist. Skipping vocabulary load.")
        return
    pattern = re.compile(r'^entry\((\w+),', re.MULTILINE)
    try:
        with open(WORDS_PL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            words_db_set = set(pattern.findall(content))
        print(f"[VOCAB] Loaded {len(words_db_set)} target words into memory.")
    except Exception as e:
        print(f"[VOCAB-ERROR] Failed to load words.pl: {e}")

def process_article_text(page_id, text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    total_tokens = len(tokens)
    if total_tokens > 0:
        local_map = {}
        for abs_pos, token in enumerate(tokens):
            if token in words_db_set:
                len_inv_pos = round(abs_pos / float(total_tokens), 4)
                loc_tuple = f"'{page_id}-{abs_pos}-{len_inv_pos}'"
                if token not in local_map:
                    local_map[token] = []
                local_map[token].append(loc_tuple)

        with location_lock:
            for token, locs in local_map.items():
                if token not in word_locations:
                    word_locations[token] = []
                word_locations[token].extend(locs)

    raw_sentences = re.split(r'[.?!]\s+', text)
    for sent in raw_sentences:
        clean_sent = sent.strip()
        if len(clean_sent) > 5:
            update_job_state(JOB_STATE["state"], JOB_STATE["progress"], JOB_STATE["total"], current_sentence=clean_sent)
            word_list = re.findall(r'\b[a-zA-Z0-9]+\b', clean_sent.lower())
            if word_list:
                with metrics_lock:
                    sentence_lengths.append(len(word_list))

            try:
                hier_fmt, flat_fmt = evaluate_sentence_via_prolog(clean_sent)
            except Exception as e:
                print(f"[PARSE-ERROR] Exception during evaluate_sentence_via_prolog: {e}")
                hier_fmt, flat_fmt = None, None

            with format_lock:
                if hier_fmt:
                    sentence_formats[hier_fmt] = sentence_formats.get(hier_fmt, 0) + 1
                if flat_fmt:
                    flat_pos_formats[flat_fmt] = flat_pos_formats.get(flat_fmt, 0) + 1

def update_words_pl_file():
    print("[VOCAB-UPDATE] Updating Trealla Prolog words.pl file with accumulated word locations...")
    if not os.path.exists(WORDS_PL_PATH):
        print("[VOCAB-UPDATE] words.pl not found. Skipping file update.")
        return
        
    with location_lock:
        current_locs = {k: list(v) for k, v in word_locations.items()}
        word_locations.clear()

    if not current_locs:
        print("[VOCAB-UPDATE] No new word locations to update.")
        return

    updated_lines = []
    entry_pattern = re.compile(r'^entry\((\w+),\s*([^,]+),\s*(\[[^\]]*\]),\s*(.+?)(?:,\s*(\[[^\]]*\]))?\)\.\s*$')

    try:
        with open(WORDS_PL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stripped = line.strip()
                match = entry_pattern.match(stripped)
                if match:
                    w, pos, plurals, desc, existing_locs_str = match.groups()
                    existing_locs = [loc.strip() for loc in existing_locs_str.strip('[]').split(',') if loc.strip()] if existing_locs_str else []
                    new_locs = current_locs.get(w, [])
                    seen = set()
                    merged_locs = []
                    for loc in existing_locs + new_locs:
                        if loc not in seen:
                            seen.add(loc)
                            merged_locs.append(loc)
                    loc_list_str = "[" + ", ".join(merged_locs) + "]"
                    updated_lines.append(f"entry({w}, {pos}, {plurals}, {desc}, {loc_list_str}).\n")
                else:
                    updated_lines.append(line)

        with open(WORDS_PL_PATH, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)
        print("[VOCAB-UPDATE] words.pl update complete with state preservation.")
    except Exception as e:
        print(f"[VOCAB-UPDATE-ERROR] {e}")

def get_total_size():
    return sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))) / (1024 ** 3)

def get_vocabulary_metrics():
    word_freqs = {}
    pos_counts = {}
    found_pos_counts = {}
    word_to_pos = {}
    
    entry_pattern = re.compile(r'^entry\((\w+),\s*([^,]+),\s*(\[[^\]]*\]),\s*(.+?)(?:,\s*(\[[^\]]*\]))?\)\.\s*$')
    if os.path.exists(WORDS_PL_PATH):
        try:
            with open(WORDS_PL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    match = entry_pattern.match(line.strip())
                    if match:
                        w, pos, plurals, desc, existing_locs_str = match.groups()
                        pos_stripped = pos.strip()
                        word_to_pos[w] = pos_stripped
                        locs = [l.strip() for l in existing_locs_str.strip('[]').split(',')] if existing_locs_str else []
                        pos_counts[pos_stripped] = pos_counts.get(pos_stripped, 0) + 1
                        if locs:
                            word_freqs[w] = word_freqs.get(w, 0) + len(locs)
        except Exception:
            pass

    with location_lock:
        for w, locs in word_locations.items():
            if locs:
                word_freqs[w] = word_freqs.get(w, 0) + len(locs)

    for w in word_freqs.keys():
        p = word_to_pos.get(w, "unknown")
        found_pos_counts[p] = found_pos_counts.get(p, 0) + 1

    found_vocab_count = len(word_freqs)
    total_vocab = len(words_db_set)
    
    sorted_words = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)
    top_ten = [{"word": w, "count": c} for w, c in sorted_words[:10]]

    with format_lock:
        sorted_formats = sorted(sentence_formats.items(), key=lambda x: x[1], reverse=True)
        top_ten_formats = [{"format": fmt, "count": cnt} for fmt, cnt in sorted_formats[:10]]
        unique_formats_count = len(sentence_formats)

        sorted_flat = sorted(flat_pos_formats.items(), key=lambda x: x[1], reverse=True)
        top_ten_flat_pos = [{"format": fmt, "count": cnt} for fmt, cnt in sorted_flat[:10]]

    with metrics_lock:
        avg_len = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0

    return {
        "vocabulary_size": total_vocab,
        "found_vocabulary_count": found_vocab_count,
        "coverage_percentage": round((found_vocab_count / total_vocab * 100), 2) if total_vocab > 0 else 0,
        "top_ten_words": top_ten,
        "total_pos_counts": pos_counts,
        "found_pos_counts": found_pos_counts,
        "top_ten_sentence_formats": top_ten_formats,
        "top_ten_flat_pos": top_ten_flat_pos,
        "average_sentence_length": round(avg_len, 2),
        "unique_sentence_formats_count": unique_formats_count
    }

def fetch_random_article_batch(session, limit=20):
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "random", "rnnamespace": "0", "rnlimit": limit, "format": "json"}
    backoff = 1
    for attempt in range(3):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 429:
                print(f"[API] Hit 429 on random batch. Backing off for {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            return r.json().get("query", {}).get("random", [])
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(backoff)
            backoff *= 2
    return []

def fetch_page_contents_batch(session, page_ids):
    if not page_ids:
        return {}
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "pageids": "|".join(str(pid) for pid in page_ids),
        "format": "json"
    }
    backoff = 1
    for attempt in range(3):
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=20)
            if r.status_code == 429:
                print(f"[API] Hit 429 Too Many Requests on batch content fetch. Backing off for {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
            results = {}
            for pid_str, page_data in pages.items():
                if "missing" in page_data:
                    continue
                revisions = page_data.get("revisions", [])
                if not revisions:
                    continue
                rev = revisions[0]
                content = rev.get("slots", {}).get("main", {}).get("*") or rev.get("*")
                if content:
                    results[int(pid_str)] = content
            return results
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(backoff)
            backoff *= 2
    return {}

def execute_download_job(n_articles):
    global is_processing
    is_processing = True
    update_job_state("PROCESSING", 0, n_articles)
    print(f"[JOB] Starting decoupled background API batch download job for target count: {n_articles}")
    try:
        load_words_pl_vocabulary()
        init_prolog_worker()
        
        downloaded = set()
        parsed_ids = set()
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                downloaded = set(line.strip() for line in f if line.strip())
        if os.path.exists(PARSED_STATE_FILE):
            with open(PARSED_STATE_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                parsed_ids = set(line.strip() for line in f if line.strip())

        print(f"[JOB] Found {len(downloaded)} downloaded and {len(parsed_ids)} parsed article IDs in state files.")

        session = requests.Session()
        total_size_gb = get_total_size()

        processed_count = 0
        with open(STATE_FILE, 'a', encoding='utf-8') as state_file_obj, open(PARSED_STATE_FILE, 'a', encoding='utf-8') as parsed_file_obj:
            while processed_count < n_articles and total_size_gb < SIZE_LIMIT_GB:
                try:
                    random_pages = fetch_random_article_batch(session, limit=20)
                except Exception as e:
                    print(f"[JOB] Error fetching random article batch from API: {e}")
                    time.sleep(1)
                    continue

                if not random_pages:
                    time.sleep(0.5)
                    continue

                unprocessed = [p for p in random_pages if str(p["id"]) not in downloaded]
                if not unprocessed:
                    continue

                page_ids = [p["id"] for p in unprocessed]
                try:
                    contents_map = fetch_page_contents_batch(session, page_ids)
                except Exception as e:
                    print(f"[JOB] Error fetching batch contents: {e}")
                    time.sleep(1)
                    continue

                for p in unprocessed:
                    if processed_count >= n_articles or total_size_gb >= SIZE_LIMIT_GB:
                        print(f"[JOB] Limit reached. Processed: {processed_count}/{n_articles}, Size: {total_size_gb:.2f}GB")
                        break

                    page_id = str(p["id"])
                    raw_text = contents_map.get(p["id"])
                    if not raw_text or re.match(r'^\s*#?REDIRECT', raw_text, re.IGNORECASE):
                        state_file_obj.write(f"{page_id}\n")
                        state_file_obj.flush()
                        downloaded.add(page_id)
                        update_job_state("PROCESSING", processed_count, n_articles)
                        continue

                    try:
                        parsed_text = mwparserfromhell.parse(raw_text).strip_code()
                    except Exception:
                        parsed_text = raw_text

                    if re.match(r'^\s*#?REDIRECT', parsed_text, re.IGNORECASE):
                        state_file_obj.write(f"{page_id}\n")
                        state_file_obj.flush()
                        downloaded.add(page_id)
                        update_job_state("PROCESSING", processed_count, n_articles)
                        continue

                    cleaned_text = "\n\n".join(b.strip() for b in parsed_text.split('\n') if len(b.strip()) > 20)
                    if not cleaned_text.strip() or len(cleaned_text.split()) < 10:
                        state_file_obj.write(f"{page_id}\n")
                        state_file_obj.flush()
                        downloaded.add(page_id)
                        update_job_state("PROCESSING", processed_count, n_articles)
                        continue

                    out_path = os.path.join(OUTPUT_DIR, f"INFO_{page_id}.txt")
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)

                    # Mark as downloaded instantly
                    state_file_obj.write(f"{page_id}\n")
                    state_file_obj.flush()
                    downloaded.add(page_id)
                    processed_count += 1
                    total_size_gb += os.path.getsize(out_path) / (1024 ** 3)
                    update_job_state("PROCESSING", processed_count, n_articles)

                    # Decoupled Asynchronous Parsing: Process text in a separate daemon thread
                    if page_id not in parsed_ids:
                        def background_parse(pid, text_content):
                            try:
                                process_article_text(pid, text_content)
                                with open(PARSED_STATE_FILE, 'a', encoding='utf-8') as pf:
                                    pf.write(f"{pid}\n")
                                    pf.flush()
                                parsed_ids.add(pid)
                            except Exception as parse_err:
                                print(f"[PARSE-BACKGROUND-ERROR] Failed parsing {pid}: {parse_err}")

                        threading.Thread(target=background_parse, args=(page_id, cleaned_text), daemon=True).start()

                    print(f"[JOB] Successfully ingested article {processed_count}/{n_articles} (ID: {page_id})")

                time.sleep(0.1)

        update_words_pl_file()
        update_job_state("COMPLETED", processed_count, n_articles, current_sentence="Ingestion Complete.")
        print("[JOB] Download job completed successfully!")
    except Exception as e:
        print(f"[JOB-ERROR] Exception encountered during download job:")
        traceback.print_exc()
        update_job_state("FAILED", JOB_STATE["progress"], n_articles, error=str(e))
    finally:
        is_processing = False

class DaemonHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(f"[HTTP] Incoming GET request for path: {self.path}")
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == '/status':
            update_job_state(JOB_STATE["state"], JOB_STATE["progress"], JOB_STATE["total"], JOB_STATE.get("current_sentence", ""), JOB_STATE.get("error"))
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(JOB_STATE).encode('utf-8'))
        elif path == '/words.pl':
            if os.path.exists(WORDS_PL_PATH):
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                self.end_headers()
                with open(WORDS_PL_PATH, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"words.pl not found on server.")
        elif path in ['/metrics', '/word-metrics']:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            downloaded_count = 0
            try:
                if os.path.exists(STATE_FILE):
                    with open(STATE_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                        downloaded_count = sum(1 for _ in f)
            except Exception:
                pass
            vocab_metrics = get_vocabulary_metrics()
            metrics = {
                **vocab_metrics,
                "downloaded_articles": downloaded_count,
                "disk_usage_gb": round(get_total_size(), 4),
                "size_limit_gb": SIZE_LIMIT_GB,
                "job_status": JOB_STATE["state"]
            }
            self.wfile.write(json.dumps(metrics).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        print(f"[HTTP] Incoming POST request for path: {self.path}")
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == '/download':
            content_length = int(self.headers.get('Content-Length', 0))
            try:
                n = int(json.loads(self.rfile.read(content_length).decode('utf-8')).get('n', 5))
            except Exception:
                n = 5

            print(f"[HTTP] Parsed download request target count: {n}")

            if is_processing:
                print("[HTTP] Conflict: Download job already running (409)")
                self.send_response(409)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "busy", "message": "A download job is already running."}).encode())
                return

            print("[HTTP] Spawning background thread for execute_download_job...")
            threading.Thread(target=execute_download_job, args=(n,)).start()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "n": n, "message": f"Queued download of {n} valid articles."}).encode())
        elif path == '/query':
            content_length = int(self.headers.get('Content-Length', 0))
            try:
                data = json.loads(self.rfile.read(content_length).decode('utf-8'))
                query_str = data.get('query') or data.get('goal', '')
            except Exception:
                query_str = ''

            if not query_str:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing 'query' or 'goal' in request body."}).encode('utf-8'))
                return

            try:
                if not query_str.strip().endswith('.'):
                    query_str = query_str.strip() + '.'
                
                with prolog_lock:
                    init_prolog_worker()
                    if prolog_proc and prolog_proc.poll() is None:
                        prolog_proc.stdin.write(query_str + "\n")
                        prolog_proc.stdin.flush()
                        stdout_res = prolog_proc.stdout.readline().strip()
                    else:
                        stdout_res = "Prolog worker inactive."

                response_data = {
                    "stdout": stdout_res,
                    "stderr": "",
                    "returncode": 0
                }
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            print(f"[HTTP] Unhandled POST path: {self.path} (404)")
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    load_words_pl_vocabulary()
    init_prolog_worker()
    server = ThreadingHTTPServer(('0.0.0.0', 5000), DaemonHandler)
    print("Download Daemon listening on port 5000...")
    server.serve_forever()