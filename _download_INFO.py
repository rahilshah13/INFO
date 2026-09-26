import json, os, re, shutil, subprocess, threading, time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import mwparserfromhell
import requests

HEADERS = {'User-Agent': 'WikiDataExtractor/1.0 (admin@example.com)'}
OUTPUT_DIR = "/info_txt_volume/"
STATE_FILE = os.path.join(OUTPUT_DIR, "downloaded_ids.txt")
WORDS_PL_PATH = os.path.join(OUTPUT_DIR, "words.pl")
SIZE_LIMIT_GB = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)
if not os.path.exists(WORDS_PL_PATH) and os.path.exists("words.pl"):
    shutil.copy("words.pl", WORDS_PL_PATH)

word_locations, location_lock, subprocess_lock = {}, threading.Lock(), threading.Lock()
words_db_set, is_processing = set(), False
JOB_STATE = {"state": "IDLE", "progress": 0, "total": 0, "error": None}

def load_words_pl_vocabulary():
    global words_db_set
    if os.path.exists(WORDS_PL_PATH):
        with open(WORDS_PL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            words_db_set = set(re.compile(r'^entry\((\w+),', re.MULTILINE).findall(f.read()))

def generate_ollama_embeddings(text):
    models = ["e5-large-instruct", "gemma2:2b", "nomic-embed-text"]
    for model_name in models:
        try:
            res = requests.post("http://127.0.0.1:11434/api/embeddings", json={
                "model": model_name,
                "prompt": text[:1000]
            }, timeout=10)
            if res.ok:
                embedding = res.json().get("embedding", [])
                if embedding:
                    return {"dim": len(embedding), "sample": embedding[:5], "model": model_name}
        except Exception:
            continue
    return {"dim": 0, "sample": [], "model": "none"}

def process_article_text(page_id, text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not tokens: return
    local_map = {}
    for abs_pos, token in enumerate(tokens):
        if token in words_db_set:
            local_map.setdefault(token, []).append(f"'{page_id}-{abs_pos}-{round(abs_pos / len(tokens), 4)}'")
    with location_lock:
        for k, v in local_map.items():
            word_locations.setdefault(k, []).extend(v)

def update_words_pl_file():
    if not os.path.exists(WORDS_PL_PATH): return
    with location_lock:
        current_locs = {k: list(v) for k, v in word_locations.items()}
        word_locations.clear()
    if not current_locs: return
    
    entry_pattern = re.compile(r'^entry\((\w+),\s*([^,]+),\s*(\[[^\]]*\]),\s*(.+?)(?:,\s*(\[[^\]]*\]))?\)\.\s*$')
    with open(WORDS_PL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    updated = []
    for line in lines:
        match = entry_pattern.match(line.strip())
        if match:
            w, pos, plurals, desc, existing_locs_str = match.groups()
            existing = [l.strip() for l in existing_locs_str.strip('[]').split(',') if l.strip()] if existing_locs_str else []
            merged = list(dict.fromkeys(existing + current_locs.get(w, [])))
            updated.append(f"entry({w}, {pos}, {plurals}, {desc}, [{', '.join(merged)}]).\n")
        else:
            updated.append(line)
            
    with open(WORDS_PL_PATH, 'w', encoding='utf-8') as f:
        f.writelines(updated)

def get_total_size():
    return sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))) / (1024 ** 3)

def get_article_metadata(page_id):
    pid_str = str(page_id)
    matches = []
    embedding_count = 0
    if os.path.exists(WORDS_PL_PATH):
        try:
            with open(WORDS_PL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if f"'{pid_str}-" in line:
                        embedding_count += line.count(f"'{pid_str}-")
                        matches.append(line.strip()[:100])
        except Exception:
            pass
    
    meta_path = os.path.join(OUTPUT_DIR, f"INFO_{pid_str}.json")
    embedding_tensor = {"dim": 0, "sample": [], "model": "none"}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as mf:
                embedding_tensor = json.load(mf).get("embedding", embedding_tensor)
        except Exception:
            pass

    return embedding_count, matches[:5], embedding_tensor

def search_articles_paginated(query_str="", page=1, limit=5):
    if not os.path.exists(OUTPUT_DIR):
        return {"items": [], "total": 0, "page": page, "limit": limit}
    
    all_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("INFO_") and f.endswith(".txt")]
    matched_articles = []
    
    for fname in sorted(all_files, reverse=True):
        page_id = fname.replace("INFO_", "").replace(".txt", "")
        file_path = os.path.join(OUTPUT_DIR, fname)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if query_str and query_str.lower() not in content.lower():
                continue
                
            embedding_count, entry_matches, embedding_tensor = get_article_metadata(page_id)
            matched_articles.append({
                "page_id": page_id,
                "filename": fname,
                "snippet": content[:200].replace("\n", " "),
                "embedding_count": embedding_count,
                "embedding_tensor": embedding_tensor,
                "entry_matches": entry_matches,
                "indexed_status": "fully_indexed" if embedding_count > 0 else "partial_or_pending"
            })
        except Exception:
            continue

    total = len(matched_articles)
    start = (page - 1) * limit
    end = start + limit
    paginated_items = matched_articles[start:end]

    return {
        "items": paginated_items,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": max(1, (total + limit - 1) // limit)
    }

def get_vocabulary_metrics():
    word_freqs, pos_counts, word_to_pos = {}, {}, {}
    entry_pattern = re.compile(r'^entry\((\w+),\s*([^,]+),\s*(\[[^\]]*\]),\s*(.+?)(?:,\s*(\[[^\]]*\]))?\)\.\s*$')
    if os.path.exists(WORDS_PL_PATH):
        with open(WORDS_PL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if match := entry_pattern.match(line.strip()):
                    w, pos, _, _, locs = match.groups()
                    p = pos.strip()
                    word_to_pos[w] = p
                    pos_counts[p] = pos_counts.get(p, 0) + 1
                    if locs:
                        word_freqs[w] = word_freqs.get(w, 0) + len([l.strip() for l in locs.strip('[]').split(',') if l.strip()])
    with location_lock:
        for w, locs in word_locations.items():
            if locs: word_freqs[w] = word_freqs.get(w, 0) + len(locs)
    
    found_pos_counts = {}
    for w in word_freqs:
        p = word_to_pos.get(w, "unknown")
        found_pos_counts[p] = found_pos_counts.get(p, 0) + 1
        
    found_vocab_count = len(word_freqs)
    total_vocab = len(words_db_set)
    top_ten = [{"word": w, "count": c} for w, c in sorted(word_freqs.items(), key=lambda x: x[1], reverse=True)[:10]]
    
    return {
        "vocabulary_size": total_vocab,
        "found_vocabulary_count": found_vocab_count,
        "coverage_percentage": round((found_vocab_count / total_vocab * 100), 2) if total_vocab else 0,
        "top_ten_words": top_ten,
        "total_pos_counts": pos_counts,
        "found_pos_counts": found_pos_counts
    }

def fetch_random_article_batch(session, limit=20):
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "random", "rnnamespace": "0", "rnlimit": limit, "format": "json"}
    backoff = 1
    for _ in range(3):
        r = session.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 429:
            time.sleep(backoff)
            backoff *= 2
            continue
        r.raise_for_status()
        return r.json().get("query", {}).get("random", [])
    return []

def fetch_page_contents_batch(session, page_ids):
    if not page_ids: return {}
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "prop": "revisions", "rvprop": "content", "rvslots": "main", "pageids": "|".join(map(str, page_ids)), "format": "json"}
    backoff = 1
    for _ in range(3):
        r = session.get(url, params=params, headers=HEADERS, timeout=20)
        if r.status_code == 429:
            time.sleep(backoff)
            backoff *= 2
            continue
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        return {int(pid): (rev[0].get("slots", {}).get("main", {}).get("*") or rev[0].get("*")) for pid, page in pages.items() if "missing" not in page and (rev := page.get("revisions"))}
    return {}

def execute_download_job(n_articles):
    global is_processing, JOB_STATE
    is_processing, JOB_STATE = True, {"state": "PROCESSING", "progress": 0, "total": n_articles, "error": None}
    load_words_pl_vocabulary()
    downloaded = set(open(STATE_FILE).read().splitlines()) if os.path.exists(STATE_FILE) else set()
    session = requests.Session()
    executor = ThreadPoolExecutor(max_workers=8)
    processed_count = 0
    with open(STATE_FILE, 'a') as state_file_obj:
        while processed_count < n_articles and get_total_size() < SIZE_LIMIT_GB:
            random_pages = fetch_random_article_batch(session, limit=20)
            if not random_pages:
                time.sleep(1)
                continue
            unprocessed = [p for p in random_pages if str(p["id"]) not in downloaded]
            if not unprocessed: continue
            contents_map = fetch_page_contents_batch(session, [p["id"] for p in unprocessed])
            for p in unprocessed:
                if processed_count >= n_articles or get_total_size() >= SIZE_LIMIT_GB: break
                page_id, raw_text = str(p["id"]), contents_map.get(p["id"])
                if not raw_text or re.match(r'^\s*#?REDIRECT', raw_text, re.IGNORECASE):
                    state_file_obj.write(f"{page_id}\n")
                    state_file_obj.flush()
                    downloaded.add(page_id)
                    continue
                parsed = mwparserfromhell.parse(raw_text).strip_code()
                if re.match(r'^\s*#?REDIRECT', parsed, re.IGNORECASE):
                    state_file_obj.write(f"{page_id}\n")
                    state_file_obj.flush()
                    downloaded.add(page_id)
                    continue
                cleaned = "\n\n".join(b.strip() for b in parsed.split('\n') if len(b.strip()) > 20)
                if not cleaned or len(cleaned.split()) < 10:
                    state_file_obj.write(f"{page_id}\n")
                    state_file_obj.flush()
                    downloaded.add(page_id)
                    continue
                
                embedding_tensor = generate_ollama_embeddings(cleaned)
                meta_path = os.path.join(OUTPUT_DIR, f"INFO_{page_id}.json")
                with open(meta_path, 'w', encoding='utf-8') as mf:
                    json.dump({"page_id": page_id, "embedding": embedding_tensor}, mf)

                out_path = os.path.join(OUTPUT_DIR, f"INFO_{page_id}.txt")
                with open(out_path, 'w', encoding='utf-8') as f: f.write(cleaned)
                executor.submit(process_article_text, page_id, cleaned)
                state_file_obj.write(f"{page_id}\n")
                state_file_obj.flush()
                downloaded.add(page_id)
                processed_count += 1
                JOB_STATE["progress"] = processed_count
            time.sleep(0.5)
    executor.shutdown(wait=True)
    update_words_pl_file()
    JOB_STATE["state"] = "COMPLETED"
    is_processing = False

class DaemonHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query_params = parse_qs(parsed_url.query)

        if path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(JOB_STATE).encode('utf-8'))
        elif path == '/words.pl' and os.path.exists(WORDS_PL_PATH):
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            with open(WORDS_PL_PATH, 'rb') as f: self.wfile.write(f.read())
        elif path in ('/metrics', '/word-metrics'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            downloaded_count = len(open(STATE_FILE).read().splitlines()) if os.path.exists(STATE_FILE) else 0
            metrics = {**get_vocabulary_metrics(), "downloaded_articles": downloaded_count, "disk_usage_gb": round(get_total_size(), 4), "size_limit_gb": SIZE_LIMIT_GB, "job_status": JOB_STATE["state"]}
            self.wfile.write(json.dumps(metrics).encode('utf-8'))
        elif path == '/articles/search':
            q = query_params.get('q', [''])[0]
            page = int(query_params.get('page', [1])[0])
            limit = int(query_params.get('limit', [5])[0])
            result = search_articles_paginated(query_str=q, page=page, limit=limit)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        if path == '/download':
            n = json.loads(body.decode('utf-8')).get('n', 5) if body else 5
            if is_processing:
                self.send_response(409)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "busy"}).encode())
                return
            threading.Thread(target=execute_download_job, args=(n,)).start()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "n": n}).encode())
        elif path == '/query':
            query_str = (json.loads(body.decode('utf-8')).get('query') or json.loads(body.decode('utf-8')).get('goal', '')) if body else ''
            if not query_str:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Missing query"}).encode())
                return
            if not query_str.strip().endswith('.'): query_str += '.'
            cmd = ["tpl", WORDS_PL_PATH, "-g", f"{query_str} halt."] if os.path.exists(WORDS_PL_PATH) else ["tpl", "-g", f"{query_str} halt."]
            with subprocess_lock:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"stdout": res.stdout, "stderr": res.stderr, "returncode": res.returncode}).encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    load_words_pl_vocabulary()
    ThreadingHTTPServer(('0.0.0.0', 5000), DaemonHandler).serve_forever()