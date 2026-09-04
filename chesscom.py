"""Chess.com PubAPI client. Serial requests + on-disk cache of monthly archives."""
import json, os, time, pathlib, urllib.request, urllib.error

BASE = "https://api.chess.com/pub"
# Chess.com 403s requests without a real User-Agent. Override with CHESS_UA env var.
UA = os.environ.get("CHESS_UA", "chess-improve/0.1 (github.com/jayeed17)")
CACHE = pathlib.Path("data/raw")


def _get(url, retries=5):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(2 ** attempt)
                continue
            if e.code in (404, 410):
                return None
            raise
        except urllib.error.URLError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after {retries} retries: {url}")


def profile(user):
    return _get(f"{BASE}/player/{user.lower()}")


def stats(user):
    return _get(f"{BASE}/player/{user.lower()}/stats")


def archives(user):
    d = _get(f"{BASE}/player/{user.lower()}/games/archives")
    return d["archives"] if d else []


def monthly(user, url, refresh_current=True):
    """Fetch one monthly archive. Past months are cached forever; current month re-fetched."""
    yyyy, mm = url.rstrip("/").split("/")[-2:]
    path = CACHE / user.lower() / f"{yyyy}-{mm}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    is_current = (yyyy, mm) == (time.strftime("%Y"), time.strftime("%m"))
    if path.exists() and not (is_current and refresh_current):
        return json.loads(path.read_text())
    data = _get(url) or {"games": []}
    path.write_text(json.dumps(data))
    return data


def all_games(user, since=None):
    """All finished games, oldest first. `since` = 'YYYY/MM' lower bound."""
    out = []
    for url in archives(user):
        if since and url.split("/games/")[1] < since:
            continue
        d = monthly(user, url)
        out.extend(d.get("games", []))
        print(f"  {url.split('/games/')[1]}: {len(d.get('games', []))} games", flush=True)
    return out


if __name__ == "__main__":
    import sys
    user = sys.argv[1]
    p = profile(user)
    print(p["url"], "| joined", time.strftime("%Y-%m-%d", time.gmtime(p["joined"])))
    s = stats(user)
    for k, v in s.items():
        if isinstance(v, dict) and "last" in v:
            rec = v.get("record", {})
            print(f"  {k:16} {v['last']['rating']:>5}  "
                  f"W{rec.get('win',0)}/L{rec.get('loss',0)}/D{rec.get('draw',0)}")
    g = all_games(user)
    print(f"total games: {len(g)}")
