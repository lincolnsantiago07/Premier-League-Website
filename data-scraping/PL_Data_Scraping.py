import time
import random
import re
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup, Comment
from typing import Optional

# =========================
# Main config
# =========================
BASE = "https://fbref.com"
START_URL = f"{BASE}/en/comps/9/Premier-League-Stats"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

# Esquema "original" do FBref que vamos capturar primeiro
FBREF_COLS = [
    "Player", "Nation", "Pos", "Age", "MP", "Starts", "Min",
    "Gls", "Ast", "PK", "CrdY", "CrdR", "xG", "xAG", "Team"
]

# Esquema final do PostgreSQL (sua tabela)
DB_COLS = [
    "player_name", "nation", "position", "age",
    "matches_played", "starts", "minutes_played",
    "goals", "assists", "penalties_scored",
    "yellow_cards", "red_cards",
    "expected_goals", "expected_assists",
    "team_name",
]

REQUEST_TIMEOUT = 25
DELAY_RANGE = (5.0, 9.0)
DROP_TOTAL_ROWS = True

CF_MARKERS = ("Just a moment", "cf-browser-verification", "cf-chl-", "Attention Required!")


# =========================
# Utilities
# =========================
def soup_from_html(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def is_cloudflare(text: str) -> bool:
    return any(m in text[:5000] for m in CF_MARKERS)


def fetch_html(url: str, max_retries: int = 4) -> str:
    sess = requests.Session()
    sess.headers.update(HEADERS)
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            r = sess.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code in (429, 403, 503) or is_cloudflare(r.text):
                raise RuntimeError(f"Blocked/Challenge (status={r.status_code})")
            return r.text
        except Exception as e:
            last_err = e
            # Optional fallback usando cloudscraper (se instalado)
            try:
                import importlib
                cs = importlib.import_module("cloudscraper")
                scraper = cs.create_scraper(browser={"custom": "chrome"})
                scraper.headers.update(HEADERS)
                rr = scraper.get(url, timeout=REQUEST_TIMEOUT + 10)
                if rr.status_code in (429, 403, 503) or is_cloudflare(rr.text):
                    raise RuntimeError(f"Blocked via cloudscraper (status={rr.status_code})")
                return rr.text
            except Exception:
                time.sleep((2 ** attempt) + random.uniform(0.3, 1.7))
                continue

    raise RuntimeError(f"Failed to fetch HTML from {url}. Last error: {last_err}")


def commented_soup(s: BeautifulSoup) -> Optional[BeautifulSoup]:
    comments = s.find_all(string=lambda x: isinstance(x, Comment))
    if not comments:
        return None
    block = "\n".join(c for c in comments if c and "table" in c.lower())
    return soup_from_html(block) if block.strip() else None


def pick_standard_table(s: BeautifulSoup):
    # 1) por id
    t = s.select_one("table.stats_table[id^=stats_standard]")
    if t:
        return t

    # 2) por caption + th 'player'
    for tb in s.select("table.stats_table"):
        cap = tb.find("caption")
        thead = tb.find("thead")
        has_player = bool(thead and thead.find("th", attrs={"data-stat": "player"}))
        if cap and re.search(r"standard", cap.get_text(" ", strip=True), re.I) and has_player:
            return tb

    # 3) nas seções comentadas
    cs = commented_soup(s)
    if cs:
        t = cs.select_one("table.stats_table[id^=stats_standard]")
        if t:
            return t
        for tb in cs.select("table.stats_table"):
            cap = tb.find("caption")
            thead = tb.find("thead")
            has_player = bool(thead and thead.find("th", attrs={"data-stat": "player"}))
            if cap and re.search(r"standard", cap.get_text(" ", strip=True), re.I) and has_player:
                return tb

    # 4) fallback
    return s.find("table", class_="stats_table")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.droplevel(0)
        except Exception:
            df.columns = [
                " ".join([str(x) for x in col if str(x) != "nan"]).strip()
                for col in df.columns.values
            ]
    cols = [str(c).strip() for c in df.columns]
    seen, new_cols = {}, []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}.{seen[c]}")
    df.columns = new_cols
    return df


def pick_one_series(df: pd.DataFrame, key: str) -> pd.Series:
    if key in df.columns:
        col = df[key]
        return col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col

    variants = [c for c in df.columns if c == key or c.startswith(f"{key}.")]
    if variants:
        col = df[variants[0]]
        return col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col

    low_map = {str(c).lower(): c for c in df.columns}
    if key.lower() in low_map:
        col = df[low_map[key.lower()]]
        return col.iloc[:, 0] if isinstance(col, pd.DataFrame) else col

    return pd.Series([pd.NA] * len(df), index=df.index)


# ---------- Limpezas específicas p/ bater com DB ----------
NUM_INT_FBREF = ["MP", "Starts", "Age", "Min", "Gls", "Ast", "PK", "CrdY", "CrdR"]
NUM_FLOAT_FBREF = ["xG", "xAG"]

def _clean_pos(val):
    if pd.isna(val):
        return pd.NA
    return str(val).split(",", 1)[0].strip()

def _clean_age(val):
    if pd.isna(val):
        return pd.NA
    m = re.search(r"(\d+)", str(val))
    return int(m.group(1)) if m else pd.NA

def _clean_nation(val):
    if pd.isna(val):
        return pd.NA
    s = " ".join(str(val).split())
    m = re.match(r"^([a-z]{2})\s+([A-Z]{3})$", s)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return s

def enforce_fbref_schema_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strings base
    for c in ["Player", "Nation", "Pos", "Team"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Limpezas
    if "Pos" in df.columns:
        df["Pos"] = df["Pos"].map(_clean_pos)
    if "Age" in df.columns:
        df["Age"] = df["Age"].map(_clean_age)
    if "Nation" in df.columns:
        df["Nation"] = df["Nation"].map(_clean_nation)

    # Remover separador de milhar em Min
    if "Min" in df.columns:
        df["Min"] = df["Min"].astype(str).str.replace(",", "", regex=False)

    # Casts
    for c in NUM_INT_FBREF:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in NUM_FLOAT_FBREF:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df
# ----------------------------------------------------------


def to_fbref_schema(df: pd.DataFrame, team_name: str) -> pd.DataFrame:
    """
    Produz o dataframe no esquema FBREF_COLS.
    """
    df = normalize_columns(df)
    out = pd.DataFrame(index=df.index)
    mapping = {
        "Player": "Player", "Nation": "Nation", "Pos": "Pos", "Age": "Age",
        "MP": "MP", "Starts": "Starts", "Min": "Min",
        "Gls": "Gls", "Ast": "Ast", "PK": "PK",
        "CrdY": "CrdY", "CrdR": "CrdR",
        "xG": "xG", "xAG": "xAG",
    }
    for col in FBREF_COLS:
        out[col] = team_name if col == "Team" else pick_one_series(df, mapping.get(col, col))

    # Remove linhas inválidas/agrupadas
    out = out[out["Player"].notna() & out["Player"].astype(str).str.strip().ne("")]
    if DROP_TOTAL_ROWS:
        out = out[~out["Player"].str.contains(r"Total|Squad|Opposition", case=False, na=False)]

    # Enforça tipos/limpeza
    out = enforce_fbref_schema_types(out)

    return out[FBREF_COLS]


def fbref_to_db_schema(df_fb: pd.DataFrame) -> pd.DataFrame:
    """
    Converte do esquema FBref para o esquema do PostgreSQL e aplica tipos finais.
    """
    df = df_fb.rename(columns={
        "Player": "player_name",
        "Nation": "nation",
        "Pos": "position",
        "Age": "age",
        "MP": "matches_played",
        "Starts": "starts",
        "Min": "minutes_played",
        "Gls": "goals",
        "Ast": "assists",
        "PK": "penalties_scored",
        "CrdY": "yellow_cards",
        "CrdR": "red_cards",
        "xG": "expected_goals",
        "xAG": "expected_assists",
        "Team": "team_name",
    })

    # Tipos finais para bater com o PostgreSQL
    int_cols = [
        "age", "matches_played", "starts", "minutes_played",
        "goals", "assists", "penalties_scored", "yellow_cards", "red_cards"
    ]
    float_cols = ["expected_goals", "expected_assists"]
    str_cols = ["player_name", "nation", "position", "team_name"]

    # Strings
    for c in str_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # minutes_played -> inteiro (arredonda .0 e remove NaNs)
    if "minutes_played" in df.columns:
        df["minutes_played"] = pd.to_numeric(df["minutes_played"], errors="coerce").round(0)

    # Inteiros (como Int64 para permitir NULL na exportação CSV)
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(0).astype("Int64")

    # Floats
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ordena colunas conforme sua tabela
    return df[DB_COLS]


# =========================
# Main pipeline
# =========================
def main():
    # 1) liga -> links de squads
    league_html = fetch_html(START_URL)
    league_soup = soup_from_html(league_html)

    league_tables = league_soup.find_all("table", class_="stats_table")
    if not league_tables:
        cs = commented_soup(league_soup)
        if cs:
            league_tables = cs.find_all("table", class_="stats_table")
    if not league_tables:
        raise RuntimeError("No 'stats_table' found on the league page.")

    squads_table = league_tables[0]
    links = [a.get("href") for a in squads_table.find_all("a") if a.get("href")]
    team_urls = [f"{BASE}{l}" for l in dict.fromkeys(links) if "/squads/" in l]
    if not team_urls:
        raise RuntimeError("No '/squads/' links found.")

    # 2) extrai tabela 'Standard' e projeta -> FBref -> DB schema
    parts = []
    for idx, url in enumerate(team_urls, 1):
        team_slug = url.rstrip("/").split("/")[-1].replace("-Stats", "")
        team_name = team_slug  # simples e robusto; pode-se melhorar lendo o <h1> se desejar
        print(f"[{idx}/{len(team_urls)}] {team_name}")

        html = fetch_html(url)
        s = soup_from_html(html)

        tag = pick_standard_table(s)
        if tag is None:
            print(f"Warning: 'Standard' table not found for {team_name}. Skipping...")
            time.sleep(random.uniform(*DELAY_RANGE))
            continue

        df_raw = pd.read_html(StringIO(str(tag)))[0]
        df_fb = to_fbref_schema(df_raw, team_name)
        df_db = fbref_to_db_schema(df_fb)
        parts.append(df_db)

        time.sleep(random.uniform(*DELAY_RANGE))

    if not parts:
        raise RuntimeError("No team tables were processed successfully.")

    final_df = pd.concat(parts, ignore_index=True)

    # CSV pronto para COPY (HEADER, sem 'NaN' literal)
    final_df.to_csv(
        "prem_stats.csv",
        index=False,
        encoding="utf-8",
        na_rep=""  # células vazias viram NULL no COPY CSV
    )
    print("File 'prem_stats.csv' generated successfully.")
    print("Shape:", final_df.shape)


if __name__ == "__main__":
    main()


"""
Sql for table create

CREATE TABLE IF NOT EXISTS premier_league_player_stats (
  player_name         VARCHAR(100) NOT NULL,
  nation              VARCHAR(50),
  position            VARCHAR(50),
  age                 INTEGER,

  matches_played      INTEGER,
  starts              INTEGER,
  minutes_played      INTEGER,

  goals               INTEGER,
  assists             INTEGER,
  penalties_scored    INTEGER,
  yellow_cards        INTEGER,
  red_cards           INTEGER,

  expected_goals      DOUBLE PRECISION,
  expected_assists    DOUBLE PRECISION,

  team_name           VARCHAR(100),

);

"""
