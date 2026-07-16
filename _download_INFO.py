import requests, bz2, mwparserfromhell, os, random

DATA_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2"
INDEX_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream-index.txt.bz2"
HEADERS = {'User-Agent': 'WikiDataExtractor/1.0'}
OUTPUT_DIR = "../info_txt_volume/"
STATE_FILE = "downloaded_ids.txt"
SIZE_LIMIT_GB = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_total_size():
    return sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR)) / (1024**3)

def get_page_ids():
    with requests.get(INDEX_URL, headers=HEADERS, stream=True) as r:
        return list({line.decode(errors='ignore').split(':')[1] for line in r.iter_lines() if len(line.decode().split(':')) >= 3})

def fetch_text(page_id, offset):
    headers = {**HEADERS, 'Range': f'bytes={offset}-{offset + 2000000}'}
    r = requests.get(DATA_URL, headers=headers)
    content = r.content.decode(errors='ignore')
    tag = f"<id>{page_id}</id>"
    if tag in content:
        text = content.split(tag)[1].split("<text")[1].split("</text>")[0].split(">", 1)[1]
        return mwparserfromhell.parse(text).strip_code()

def run():
    downloaded = set(open(STATE_FILE).read().splitlines()) if os.path.exists(STATE_FILE) else set()
    all_ids = get_page_ids()
    random.shuffle(all_ids)
    
    with open(STATE_FILE, 'a') as state:
        for page_id in all_ids:
            if page_id in downloaded or get_total_size() >= SIZE_LIMIT_GB: continue
            offset = next((int(line.split(':')[0]) for line in requests.get(INDEX_URL, headers=HEADERS).text.splitlines() if line.split(':')[1] == page_id), None)
            if not offset: continue
            
            text = fetch_text(page_id, offset)
            if text:
                with open(os.path.join(OUTPUT_DIR, f"INFO_{page_id}.txt"), 'w', encoding='utf-8') as f:
                    f.write("\n\n".join([b.strip() for b in text.split('\n') if len(b.strip()) > 20]))
                state.write(f"{page_id}\n")
                downloaded.add(page_id)
                print(f"Downloaded {page_id}. Size: {get_total_size():.2f}GB")

if __name__ == "__main__":
    run()
