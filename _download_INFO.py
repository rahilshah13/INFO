import requests, mwparserfromhell, os, random, re, threading, json, shutil, traceback, time, subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from concurrent.futures import ThreadPoolExecutor

HEADERS = {'User-Agent': 'WikiDataExtractor/1.0 (admin@example.com)'}

# Point everything directly to the mounted Docker volume container path
OUTPUT_DIR = "/info_txt_volume/"
STATE_FILE = os.path.join(OUTPUT_DIR, "downloaded_ids.txt")
WORDS_PL_PATH = os.path.join(OUTPUT_DIR, "words.pl")
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
location_lock = threading.Lock()
subprocess_lock = threading.Lock()  # Protect concurrent tpl subprocess invocations
words_db_set = set()
is_processing = False
JOB_STATE = {"state": "IDLE", "progress": 0, "total": 0, "error": None}

def load_words_pl_vocabulary():
    global words_db_set
    print("[VOCAB] Loading Trealla Prolog words.pl vocabulary set...")
    if not os.path.exists(WORDS_PL_PATH):
        print("[VOCAB] words.pl path does not exist. Skipping vocabulary load.")
        return
    pattern = re.compile(r'^entry\((\w+),', re.MULTILINE)
    with open(WORDS_PL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        words_db_set = set(pattern.findall(content))
    print(f"[VOCAB] Loaded {len(words_db_set)} target words into memory.")

def process_article_text(page_id, text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    total_tokens = len(tokens)
    if total_tokens == 0:
        return

    local_map = {}
    for abs_pos, token in enumerate(tokens):
        if token in words_db_set:
            len_inv_pos = round(abs_pos / float(total_tokens), 4)
            # Wrap location in single quotes so Prolog treats it as an atom instead of an arithmetic expression
            loc_tuple = f"'{page_id}-{abs_pos}-{len_inv_pos}'"
            if token not in local_map:
                local_map[token] = []
            local_map[token].append(loc_tuple)

    with location_lock:
        for token, locs in local_map.items():
            if token not in word_locations:
                word_locations[token] = []
            word_locations[token].extend(locs)

def update_words_pl_file():
    print("[VOCAB-UPDATE] Updating Trealla Prolog words.pl file with accumulated word locations...")
    if not os.path.exists(WORDS_PL_PATH):
        print("[VOCAB-UPDATE] words.pl not found. Skipping file update.")
        return
        
    with location_lock:
        current_locs = {k: list(v) for k, v in word_locations.items()}
        word_locations.clear()  # Prevent unbounded memory accumulation/leak

    if not current_locs:
        print("[VOCAB-UPDATE] No new word locations to update.")
        return

    updated_lines = []
    
    # Robust regex matching both 4-argument and already-updated 5-argument Prolog entry facts:
    entry_pattern = re.compile(r'^entry\((\w+),\s*([^,]+),\s*(\[[^\]]*\]),\s*(.+?)(?:,\s*(\[[^\]]*\]))?\)\.\s*$')

    with open(WORDS_PL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            match = entry_pattern.match(stripped)
            if match:
                w, pos, plurals, desc, existing_locs_str = match.groups()
                
                existing_locs = []
                if existing_locs_str:
                    existing_locs = [loc.strip() for loc in existing_locs_str.strip('[]').split(',') if loc.strip()]
                
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

def get_total_size():
    return sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))) / (1024 ** 3)

def get_vocabulary_metrics():
    word_freqs = {}
    pos_counts = {}
    found_pos_counts = {}
    word_to_pos = {}
    
    entry_pattern = re.compile(r'^entry\((\w+),\s*([^,]+),\s*(\[[^\]]*\]),\s*(.+?)(?:,\s*(\[[^\]]*\]))?\)\.\s*$')
    
    # 1. Parse words.pl if it exists
    if os.path.exists(WORDS_PL_PATH):
        try:
            with open(WORDS_PL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    match = entry_pattern.match(line.strip())
                    if match:
                        w, pos, plurals, desc, existing_locs_str = match.groups()
                        pos_stripped = pos.strip()
                        word_to_pos[w] = pos_stripped
                        
                        locs = []
                        if existing_locs_str:
                            locs = [l.strip() for l in existing_locs_str.strip('[]').split(',') if l.strip()]
                        
                        pos_counts[pos_stripped] = pos_counts.get(pos_stripped, 0) + 1
                        
                        if locs:
                            word_freqs[w] = word_freqs.get(w, 0) + len(locs)
        except Exception:
            pass

    # 2. Add in-memory word_locations if any
    with location_lock:
        for w, locs in word_locations.items():
            if locs:
                word_freqs[w] = word_freqs.get(w, 0) + len(locs)

    # Calculate found counts per part of speech
    for w in word_freqs.keys():
        p = word_to_pos.get(w, "unknown")
        found_pos_counts[p] = found_pos_counts.get(p, 0) + 1

    found_vocab_count = len(word_freqs)
    total_vocab = len(words_db_set)
    
    # Sort by frequency descending, take top 10
    sorted_words = sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)
    top_ten = sorted_words[:10]
    top_ten_table = [{"word": w, "count": c} for w, c in top_ten]

    return {
        "vocabulary_size": total_vocab,
        "found_vocabulary_count": found_vocab_count,
        "coverage_percentage": round((found_vocab_count / total_vocab * 100), 2) if total_vocab > 0 else 0,
        "top_ten_words": top_ten_table,
        "total_pos_counts": pos_counts,
        "found_pos_counts": found_pos_counts
    }

def fetch_random_article_batch(session, limit=20):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "random",
        "rnnamespace": "0",
        "rnlimit": limit,
        "format": "json"
    }
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
            data = r.json()
            return data.get("query", {}).get("random", [])
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
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
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
    global is_processing, JOB_STATE
    is_processing = True
    JOB_STATE = {"state": "PROCESSING", "progress": 0, "total": n_articles, "error": None}
    print(f"[JOB] Starting background API batch download job for target count: {n_articles}")
    try:
        load_words_pl_vocabulary()
        
        downloaded = set(open(STATE_FILE).read().splitlines()) if os.path.exists(STATE_FILE) else set()
        print(f"[JOB] Found {len(downloaded)} previously downloaded article IDs in state file.")

        session = requests.Session()
        total_size_gb = get_total_size()
        executor = ThreadPoolExecutor(max_workers=8)

        processed_count = 0
        with open(STATE_FILE, 'a') as state_file_obj:
            while processed_count < n_articles and total_size_gb < SIZE_LIMIT_GB:
                try:
                    random_pages = fetch_random_article_batch(session, limit=20)
                except Exception as e:
                    print(f"[JOB] Error fetching random article batch from API: {e}")
                    time.sleep(3)
                    continue

                if not random_pages:
                    time.sleep(1)
                    continue

                unprocessed = [p for p in random_pages if str(p["id"]) not in downloaded]
                if not unprocessed:
                    continue

                page_ids = [p["id"] for p in unprocessed]
                try:
                    contents_map = fetch_page_contents_batch(session, page_ids)
                except Exception as e:
                    print(f"[JOB] Error fetching batch contents: {e}")
                    time.sleep(3)
                    continue

                for p in unprocessed:
                    if processed_count >= n_articles or total_size_gb >= SIZE_LIMIT_GB:
                        print(f"[JOB] Limit reached. Processed: {processed_count}/{n_articles}, Size: {total_size_gb:.2f}GB")
                        break

                    page_id = str(p["id"])
                    raw_text = contents_map.get(p["id"])
                    
                    if not raw_text:
                        state_file_obj.write(f"{page_id}\n")
                        state_file_obj.flush()
                        downloaded.add(page_id)
                        continue

                    if re.match(r'^\s*#?REDIRECT', raw_text, re.IGNORECASE):
                        state_file_obj.write(f"{page_id}\n")
                        state_file_obj.flush()
                        downloaded.add(page_id)
                        continue

                    parsed_text = mwparserfromhell.parse(raw_text).strip_code()
                    if re.match(r'^\s*#?REDIRECT', parsed_text, re.IGNORECASE):
                        state_file_obj.write(f"{page_id}\n")
                        state_file_obj.flush()
                        downloaded.add(page_id)
                        continue

                    cleaned_text = "\n\n".join(b.strip() for b in parsed_text.split('\n') if len(b.strip()) > 20)
                    if not cleaned_text.strip() or len(cleaned_text.split()) < 10:
                        state_file_obj.write(f"{page_id}\n")
                        state_file_obj.flush()
                        downloaded.add(page_id)
                        continue

                    out_path = os.path.join(OUTPUT_DIR, f"INFO_{page_id}.txt")
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)

                    executor.submit(process_article_text, page_id, cleaned_text)

                    total_size_gb += os.path.getsize(out_path) / (1024 ** 3)
                    state_file_obj.write(f"{page_id}\n")
                    state_file_obj.flush()
                    downloaded.add(page_id)
                    processed_count += 1
                    JOB_STATE["progress"] = processed_count
                    print(f"[JOB] Successfully saved and processed full article {processed_count}/{n_articles} (ID: {page_id})")

                time.sleep(0.5)

        print("[JOB] Shutting down thread pool executor...")
        executor.shutdown(wait=True)
        update_words_pl_file()
        JOB_STATE["state"] = "COMPLETED"
        print("[JOB] Download job completed successfully!")
    except Exception as e:
        print(f"[JOB-ERROR] Exception encountered during download job:")
        traceback.print_exc()
        JOB_STATE["state"] = "FAILED"
        JOB_STATE["error"] = str(e)
    finally:
        is_processing = False

class DaemonHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(f"[HTTP] Incoming GET request for path: {self.path}")
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        if path == '/status':
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
                    file_data = f.read()
                self.wfile.write(file_data)
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"words.pl not found on server.")
        elif path == '/metrics' or path == '/word-metrics':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()

            downloaded_count = 0
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r') as f:
                    downloaded_count = len(f.read().splitlines())

            vocab_metrics = get_vocabulary_metrics()

            metrics = {
                "vocabulary_size": vocab_metrics["vocabulary_size"],
                "found_vocabulary_count": vocab_metrics["found_vocabulary_count"],
                "coverage_percentage": vocab_metrics["coverage_percentage"],
                "top_ten_words": vocab_metrics["top_ten_words"],
                "found_pos_counts": vocab_metrics["found_pos_counts"],
                "total_pos_counts": vocab_metrics["total_pos_counts"],
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
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode('utf-8'))
                n = int(data.get('n', 5))
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
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode('utf-8'))
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
                
                goal_arg = f"{query_str} halt."
                cmd = ["tpl", WORDS_PL_PATH, "-g", goal_arg] if os.path.exists(WORDS_PL_PATH) else ["tpl", "-g", goal_arg]
                
                # Protect execution of the tpl subprocess with a thread lock to prevent concurrent collisions
                with subprocess_lock:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                
                response_data = {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
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

def run_daemon():
    server = ThreadingHTTPServer(('0.0.0.0', 5000), DaemonHandler)
    print("Download Daemon listening on port 5000...")
    server.serve_forever()

if __name__ == "__main__":
    load_words_pl_vocabulary()
    run_daemon()