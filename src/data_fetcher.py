"""
data_fetcher.py
---------------
Holdings 로딩 + 가격 수집 + 환율 변환 통합 모듈

데이터 흐름:
  Google Sheets (또는 holdings.csv)
    → 실시간 가격 (yfinance / pykrx)
    → 실시간 환율 (yfinance KRW=X)
    → 통합 포트폴리오 스냅샷 DataFrame 반환

매매 후 업데이트:
  Google Sheets에서 shares 수정 (또는 새 행 추가)
  → 코드 변경 없이 자동 반영
"""

import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ── 경로 설정
ROOT = Path(__file__).resolve().parent.parent
HOLDINGS_CSV = ROOT / "data" / "holdings.csv"


# ══════════════════════════════════════════════════════════════════════════════
# 1. HOLDINGS 로딩
#    우선순위: Google Sheets → 로컬 CSV 폴백
# ══════════════════════════════════════════════════════════════════════════════

def load_holdings_from_sheets() -> pd.DataFrame:
    """
    Google Sheets에서 holdings 로드.
    .env에 GOOGLE_SHEETS_URL, GOOGLE_SHEETS_KEY_PATH 필요.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        key_path = os.getenv("GOOGLE_SHEETS_KEY_PATH", "credentials.json")
        sheet_url = os.getenv("GOOGLE_SHEETS_URL", "")

        if not sheet_url:
            raise ValueError("GOOGLE_SHEETS_URL not set in .env")

        creds = Credentials.from_service_account_file(key_path, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_url(sheet_url).sheet1
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        print(f"✓ Google Sheets에서 {len(df)}개 포지션 로드")
        return df

    except Exception as e:
        print(f"⚠ Google Sheets 로드 실패 ({e}) → CSV 폴백")
        return None


def load_holdings(use_sheets: bool = True) -> pd.DataFrame:
    """
    Holdings 로드 (Google Sheets 우선, CSV 폴백).

    Returns
    -------
    pd.DataFrame
        컬럼: ticker, name, shares, avg_cost, currency,
               category, sub_category, market
    """
    df = None

    if use_sheets:
        df = load_holdings_from_sheets()

    if df is None:
        df = pd.read_csv(HOLDINGS_CSV)
        print(f"✓ CSV에서 {len(df)}개 포지션 로드")

    # 타입 정리
    df["shares"]   = pd.to_numeric(df["shares"],   errors="coerce").fillna(0)
    df["avg_cost"] = pd.to_numeric(df["avg_cost"],  errors="coerce").fillna(0)
    df["currency"] = df["currency"].str.upper()

    # shares = 0인 청산 종목 제거
    df = df[df["shares"] > 0].reset_index(drop=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. 환율 (USD/KRW)
#    yfinance에서 실시간으로 가져옴 → 고정값 오차 제거
# ══════════════════════════════════════════════════════════════════════════════

def get_usdkrw() -> float:
    """
    실시간 USD/KRW 환율 반환.
    yfinance 실패 시 .env의 USDKRW_FALLBACK 사용.
    """
    try:
        rate = yf.Ticker("KRW=X").fast_info["last_price"]
        if rate and rate > 0:
            print(f"✓ 환율: 1 USD = {rate:,.2f} KRW")
            return float(rate)
    except Exception:
        pass

    fallback = float(os.getenv("USDKRW_FALLBACK", 1320.0))
    print(f"⚠ 환율 실시간 조회 실패 → 폴백: {fallback:,.2f}")
    return fallback


# ══════════════════════════════════════════════════════════════════════════════
# 3. 현재가 수집
#    - 미국주식/ETF/TSX: yfinance
#    - 한국주식 (KRX):   pykrx (장마감 후 기준가)
# ══════════════════════════════════════════════════════════════════════════════

def get_current_prices_us(tickers: list) -> dict:
    """yfinance로 미국 주식 + ETF + TSX 현재가 일괄 수집."""
    if not tickers:
        return {}

    prices = {}
    try:
        data = yf.download(tickers, period="2d", auto_adjust=True, progress=False)
        close = data["Close"] if "Close" in data else data

        if isinstance(close, pd.Series):
            last_price = close.dropna().iloc[-1]
            prices[tickers[0]] = float(last_price)
        else:
            for t in tickers:
                if t in close.columns:
                    series = close[t].dropna()
                    if not series.empty:
                        prices[t] = float(series.iloc[-1])
    except Exception as e:
        print(f"⚠ yfinance 일괄 조회 오류: {e}")
        for t in tickers:
            try:
                info = yf.Ticker(t).fast_info
                prices[t] = float(info["last_price"])
            except Exception:
                print(f"  ✗ {t} 가격 조회 실패")

    return prices


def get_current_prices_kr(tickers_ks: list) -> dict:
    """
    pykrx로 한국 주식 현재가 수집.
    ticker 형식: '005930.KS' → pykrx용 '005930'으로 변환.
    """
    if not tickers_ks:
        return {}

    prices = {}
    try:
        from pykrx import stock as krx

        today     = datetime.now().strftime("%Y%m%d")
        from_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")

        for ticker_full in tickers_ks:
            code = ticker_full.replace(".KS", "").replace(".KQ", "")
            try:
                df = krx.get_market_ohlcv(from_date, today, code)
                if not df.empty:
                    prices[ticker_full] = float(df["종가"].iloc[-1])
            except Exception as e:
                print(f"  ✗ {ticker_full} pykrx 조회 실패: {e}")

    except ImportError:
        print("⚠ pykrx 미설치 → yfinance로 KRX 종목 재시도")
        prices.update(get_current_prices_us(tickers_ks))

    return prices


# ══════════════════════════════════════════════════════════════════════════════
# 4. 히스토리컬 가격 (Factor Model / Risk 분석용)
# ══════════════════════════════════════════════════════════════════════════════

def get_historical_prices(
    tickers: list,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    여러 종목의 일별 종가 히스토리 반환.

    Parameters
    ----------
    tickers  : 조회할 ticker 리스트
    period   : '1y', '2y', '3y', '5y' 등
    interval : '1d', '1wk', '1mo'

    Returns
    -------
    pd.DataFrame  index=날짜, columns=ticker, 값=종가(원본 통화)
    """
    us_tickers = [t for t in tickers if ".KS" not in t and ".KQ" not in t]
    kr_tickers = [t for t in tickers if ".KS" in t or ".KQ" in t]

    frames = []

    if us_tickers:
        try:
            raw = yf.download(
                us_tickers, period=period, interval=interval,
                auto_adjust=True, progress=False
            )
            close = raw["Close"] if "Close" in raw else raw
            if isinstance(close, pd.Series):
                close = close.to_frame(name=us_tickers[0])
            frames.append(close)
        except Exception as e:
            print(f"⚠ 히스토리 조회 오류 (US): {e}")

    if kr_tickers:
        try:
            raw_kr = yf.download(
                kr_tickers, period=period, interval=interval,
                auto_adjust=True, progress=False
            )
            close_kr = raw_kr["Close"] if "Close" in raw_kr else raw_kr
            if isinstance(close_kr, pd.Series):
                close_kr = close_kr.to_frame(name=kr_tickers[0])
            frames.append(close_kr)
        except Exception as e:
            print(f"⚠ 히스토리 조회 오류 (KR): {e}")

    if not frames:
        return pd.DataFrame()

    prices = pd.concat(frames, axis=1)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.ffill().dropna(how="all")

    print(f"✓ 히스토리 데이터: {len(prices)}일 × {len(prices.columns)}종목")
    return prices


# ══════════════════════════════════════════════════════════════════════════════
# 5. 포트폴리오 스냅샷 (핵심 함수)
#    holdings + 현재가 + 환율을 합쳐서
#    분석에 바로 쓸 수 있는 DataFrame 반환
# ══════════════════════════════════════════════════════════════════════════════

def build_portfolio_snapshot(use_sheets: bool = True) -> pd.DataFrame:
    """
    전체 포트폴리오의 현재 상태를 하나의 DataFrame으로 반환.

    Returns
    -------
    pd.DataFrame 컬럼:
        ticker, name, shares, avg_cost, currency,
        category, sub_category, market,
        current_price,        # 원본 통화 가격
        current_price_usd,    # USD 환산가
        current_price_krw,    # KRW 환산가
        value_usd,            # 평가금 (USD)
        value_krw,            # 평가금 (KRW)
        cost_krw,             # 매입금 (KRW)
        pnl_krw,              # 손익 (KRW)
        pnl_pct,              # 수익률 (%)
        weight                # 포트폴리오 비중 (%)
    """
    holdings = load_holdings(use_sheets=use_sheets)
    usdkrw   = get_usdkrw()

    kr_mask = holdings["currency"] == "KRW"
    us_mask = holdings["currency"] == "USD"

    kr_tickers = holdings.loc[kr_mask, "ticker"].tolist()
    us_tickers = holdings.loc[us_mask, "ticker"].tolist()

    print("\n📡 현재가 수집 중...")
    prices_kr  = get_current_prices_kr(kr_tickers)
    prices_us  = get_current_prices_us(us_tickers)
    all_prices = {**prices_kr, **prices_us}

    rows = []
    for _, row in holdings.iterrows():
        ticker = row["ticker"]
        price  = all_prices.get(ticker, np.nan)
        is_kr  = row["currency"] == "KRW"

        price_krw = price if is_kr else (price * usdkrw if pd.notna(price) else np.nan)
        price_usd = (price / usdkrw if is_kr else price) if pd.notna(price) else np.nan

        shares   = row["shares"]
        avg_cost = row["avg_cost"]

        value_usd = shares * price_usd if pd.notna(price_usd) else np.nan
        value_krw = shares * price_krw if pd.notna(price_krw) else np.nan
        cost_krw  = shares * avg_cost * (1 if is_kr else usdkrw)

        pnl_krw = value_krw - cost_krw if pd.notna(value_krw) else np.nan
        pnl_pct = (pnl_krw / cost_krw * 100) if (cost_krw > 0 and pd.notna(pnl_krw)) else np.nan

        rows.append({
            "ticker":            ticker,
            "name":              row["name"],
            "shares":            shares,
            "avg_cost":          avg_cost,
            "currency":          row["currency"],
            "category":          row["category"],
            "sub_category":      row["sub_category"],
            "market":            row["market"],
            "current_price":     price,
            "current_price_usd": price_usd,
            "current_price_krw": price_krw,
            "value_usd":         value_usd,
            "value_krw":         value_krw,
            "cost_krw":          cost_krw,
            "pnl_krw":           pnl_krw,
            "pnl_pct":           pnl_pct,
        })

    snapshot = pd.DataFrame(rows)

    # 비중 계산
    total_krw         = snapshot["value_krw"].sum()
    snapshot["weight"] = snapshot["value_krw"] / total_krw * 100

    print(f"\n{'─'*50}")
    print(f"  총 평가금:  ₩{total_krw:>15,.0f}")
    print(f"  총 평가금:  ${total_krw/usdkrw:>12,.2f}")
    print(f"  포지션 수:  {len(snapshot)}개")
    print(f"  USD/KRW:    {usdkrw:,.2f}")
    print(f"{'─'*50}\n")

    return snapshot


# ══════════════════════════════════════════════════════════════════════════════
# 실행 테스트
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("  Portfolio Data Fetcher — 테스트 실행")
    print("=" * 50)

    snap = build_portfolio_snapshot(use_sheets=False)

    pd.set_option("display.float_format", "{:,.1f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)

    cols = ["ticker", "name", "shares", "current_price", "value_krw", "pnl_pct", "weight", "category"]
    print(snap[cols].sort_values("value_krw", ascending=False).to_string(index=False))
