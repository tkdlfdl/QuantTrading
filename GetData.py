
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

session = requests.Session()
session.headers.update(HEADERS)

def normalize_tickers(series):
    return (
        series.dropna()
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)   # for Yahoo Finance
        .unique()
        .tolist()
    )

def get_sp500():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = session.get(url, timeout=20)
    html.raise_for_status()

    df = pd.read_html(StringIO(html.text))[0]
    return normalize_tickers(df["Symbol"])

def get_nasdaq100():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    html = session.get(url, timeout=20)
    html.raise_for_status()

    tables = pd.read_html(StringIO(html.text))

    for df in tables:
        cols = [str(c).strip() for c in df.columns]
        if "Ticker" in cols:
            return normalize_tickers(df["Ticker"])

    raise ValueError("Nasdaq-100 ticker table not found.")

sp500 = get_sp500()
nasdaq100 = get_nasdaq100()

print("S&P 500:", len(sp500))
print("Nasdaq-100:", len(nasdaq100))
print(sp500[:10])
print(nasdaq100[:10])

print(len(sp500), len(nasdaq100))
# ticker = spx+ndx+ ['QQQ','SPY']/
# # r2k +
# ndx = get_index_tickers("nasdaq100")
# sp500 = get_index_tickers("sp500")
ticker =nasdaq100+ sp500 +['TMF', 'TLT']
# ticker = r2k +['QQQ','SPY']
ticker = set(ticker)
ticker = list(ticker)
print(len(set(ticker)))

# stock_list_mom = payload[4]['Ticker'].to_list()+['QQQ']+['SPY']+['TMF']+['']
# BONDS=['TMF']+['TLT']
stock_list_mom = ticker +['QQQ']+['SPY']+['UVXY']

stock_list_mom = ticker +['QQQ']+['SPY']+['UVXY']+['^VIX']+["2Y", "US2Y", "^FVX", "DGS2"] +["10Y", "US10Y", "^TNX", "DGS10"]
# 금융위기 포함 2000년부터
start="1997-01-01"
end=today

# 종목 다운로드
df_mom = yf.download(tickers = stock_list_mom, start=start, end=end)
# df_vix = yf.download(tickers = stock_list_vix, start=start, end=end)

# 데이터 전처리
df2_mom = df_mom.bfill(axis ='rows')
df3_mom = df2_mom.ffill(axis ='rows')
df4_mom = df3_mom.dropna(axis='columns')
df4_mom
df_ret_mom = df4_mom['Close'].copy()