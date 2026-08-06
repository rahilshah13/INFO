import requests, bz2, mwparserfromhell, os, random, re, threading, json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from concurrent.futures import ThreadPoolExecutor

DATA_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2"
INDEX_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream-index.txt.bz2"
HEADERS = {'User-Agent': 'WikiDataExtractor/1.0'}
OUTPUT_DIR = "../info_txt_volume/"
STATE_FILE = os.path.join(OUTPUT_DIR, "downloaded_ids.txt")
INDEX_CACHE = os.path.join(OUTPUT_DIR, "multistream-index.txt")
WORDS_PL_PATH = "words.pl"
SIZE_LIMIT_GB = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)
word_locations = {}
location_lock = threading.Lock()
words_db_set = set()
is_processing = False
JOB_STATE = {"state": "IDLE", "progress": 0, "total": 0}

def load_words_pl_vocabulary():
    global words_db_set
    if not os.path.exists(WORDS_PL_PATH):
        return
    pattern = re.compile(r'^entry\((\w+),', re.MULTILINE)
    with open(WORDS_PL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        words_db_set = set(pattern.findall(content))

def process_article_text(page_id, text):
    tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    total_tokens = len(tokens)
    if total_tokens == 0:
        return

    local_map = {}
    for abs_pos, token in enumerate(tokens):
        if token in words_db_set:
            len_inv_pos = round(abs_pos / float(total_tokens), 4)
            loc_tuple = f"{page_id}-{abs_pos}-{len_inv_pos}"
            if token not in local_map:
                local_map[token] = []
            local_map[token].append(loc_tuple)

    with location_lock:
        for token, locs in local_map.items():
            if token not in word_locations:
                word_locations[token] = []
            word_locations[token].extend(locs)

def update_words_pl_file():
    if not os.path.exists(WORDS_PL_PATH):
        return
    with location_lock:
        current_locs = {k: list(v) for k, v in word_locations.items()}

    updated_lines = []
    entry_pattern = re.compile(r'^entry\((\w+),\s*([^,]+),\s*(\[[^\]]*\]),\s*(.+)\)\.\s*$')

    with open(WORDS_PL_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = entry_pattern.match(line.strip())
            if match:
                w, pos, plurals, desc = match.groups()
                locs = current_locs.get(w, [])
                loc_list_str = "[" + ", ".join(locs) + "]"
                updated_lines.append(f"entry({w}, {pos}, {plurals}, {desc}, {loc_list_str}).\n")
            else:
                updated_lines.append(line)

    with open(WORDS_PL_PATH, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)

def get_total_size():
    return sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR)) / (1024 ** 3)

def ensure_index():
    if os.path.exists(INDEX_CACHE):
        return
    r = requests.get(INDEX_URL, headers=HEADERS)
    r.raise_for_status()
    decompressed = bz2.decompress(r.content)
    with open(INDEX_CACHE, 'wb') as f:
        f.write(decompressed)

def load_index():
    ensure_index()
    offset_set = set()
    page_to_offset = {}
    with open(INDEX_CACHE, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.rstrip('\n').split(':', 2)
            if len(parts) < 3:
                continue
            offset = int(parts[0])
            page_to_offset[parts[1]] = offset
            offset_set.add(offset)
    offsets = sorted(offset_set)
    return offsets, page_to_offset

def fetch_block(offset, next_offset, session):
    end = (next_offset - 1) if next_offset is not None else (offset + 200_000_000)
    headers = {**HEADERS, 'Range': f'bytes={offset}-{end}'}
    r = session.get(DATA_URL, headers=headers)
    r.raise_for_status()
    decompressor = bz2.BZ2Decompressor()
    try:
        raw = decompressor.decompress(r.content)
    except OSError:
        return None
    return raw.decode('utf-8', errors='ignore')

def extract_page_text(block_xml, page_id):
    for page_match in re.finditer(r'<page>.*?</page>', block_xml, re.DOTALL):
        page_xml = page_match.group(0)
        id_match = re.search(r'<id>(\d+)</id>', page_xml)
        if id_match and id_match.group(1) == str(page_id):
            text_match = re.search(r'<text[^>]*>(.*?)</text>', page_xml, re.DOTALL)
            if text_match:
                raw_text = text_match.group(1)
                if re.match(r'^\s*#?REDIRECT', raw_text, re.IGNORECASE):
                    return None
                parsed_text = mwparserfromhell.parse(raw_text).strip_code()
                if re.match(r'^\s*#?REDIRECT', parsed_text, re.IGNORECASE):
                    return None
                return parsed_text
    return None

def execute_download_job(n_articles):
    global is_processing, JOB_STATE
    is_processing = True
    JOB_STATE = {"state": "PROCESSING", "progress": 0, "total": n_articles}
    try:
        load_words_pl_vocabulary()
        downloaded = set(open(STATE_FILE).read().splitlines()) if os.path.exists(STATE_FILE) else set()
        offsets, page_to_offset = load_index()
        offset_index = {off: i for i, off in enumerate(offsets)}

        all_page_ids = [pid for pid in page_to_offset.keys() if pid not in downloaded]
        random.shuffle(all_page_ids)

        session = requests.Session()
        total_size_gb = get_total_size()
        executor = ThreadPoolExecutor(max_workers=8)

        processed_count = 0
        with open(STATE_FILE, 'a') as state:
            for page_id in all_page_ids:
                if processed_count >= n_articles or total_size_gb >= SIZE_LIMIT_GB:
                    break

                offset = page_to_offset[page_id]
                idx = offset_index[offset]
                next_offset = offsets[idx + 1] if idx + 1 < len(offsets) else None

                block_xml = fetch_block(offset, next_offset, session)
                if not block_xml:
                    continue
                text = extract_page_text(block_xml, page_id)
                if not text or not text.strip():
                    state.write(f"{page_id}\n")
                    downloaded.add(page_id)
                    continue

                cleaned_text = "\n\n".join(b.strip() for b in text.split('\n') if len(b.strip()) > 20)
                if not cleaned_text.strip() or re.match(r'^\s*#?REDIRECT', cleaned_text, re.IGNORECASE) or len(cleaned_text.split()) < 10:
                    state.write(f"{page_id}\n")
                    downloaded.add(page_id)
                    continue

                out_path = os.path.join(OUTPUT_DIR, f"INFO_{page_id}.txt")
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)

                executor.submit(process_article_text, page_id, cleaned_text)

                total_size_gb += os.path.getsize(out_path) / (1024 ** 3)
                state.write(f"{page_id}\n")
                downloaded.add(page_id)
                processed_count += 1
                JOB_STATE["progress"] = processed_count

        executor.shutdown(wait=True)
        update_words_pl_file()
        JOB_STATE["state"] = "COMPLETED"
    except Exception:
        JOB_STATE["state"] = "FAILED"
    finally:
        is_processing = False

class DaemonHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(JOB_STATE).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/download':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode('utf-8'))
                n = int(data.get('n', 5))
            except Exception:
                n = 5

            if is_processing:
                self.send_response(409)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "busy", "message": "A download job is already running."}).encode())
                return

            threading.Thread(target=execute_download_job, args=(n,)).start()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "started", "n": n, "message": f"Queued download of {n} valid articles."}).encode())
        else:
            self.send_response(404)
            self.end_headers()

def run_daemon():
    server = HTTPServer(('0.0.0.0', 5000), DaemonHandler)
    print("Download Daemon listening on port 5000...")
    server.serve_forever()

if __name__ == "__main__":
    run_daemon()