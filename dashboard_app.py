# -*- coding: utf-8 -*-
import os, time, io, datetime as dt
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Rufino Trader – Dashboard", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
APP_TITLE = "Rufino Trader – Painel em Tempo Real"
LOCAL_TZ = "America/Fortaleza"

# ---------------------- Caminhos no repositório ----------------------
# Agora apontam exatamente para as pastas que você citou:
CSV_A_REPO_PATH = "dashboard_app/log/trader.csv"      # diário detalhado (novo ou antigo)
CSV_B_REPO_PATH = "dashboard_app/data/trade_log.csv"  # legado simples

# Também tentamos estes caminhos locais (para rodar em dev sem GitHub Raw):
CSV_A_LOCAL = os.path.join("dashboard_app", "log", "trader.csv")
CSV_B_LOCAL = os.path.join("dashboard_app", "data", "trade_log.csv")

# ---------------------- Config do GitHub Raw (opcional) ----------------------
# Informe pelo secrets ou env para ler direto do GitHub (raw.githubusercontent.com)
REPO_SLUG = st.secrets.get("repo_slug", os.getenv("RT_REPO_SLUG", "")).strip()   # ex: "alexrufino/meu-repo"
REPO_BRANCH = st.secrets.get("repo_branch", os.getenv("RT_BRANCH", "main")).strip()
GITHUB_TOKEN = st.secrets.get("github_token", os.getenv("GITHUB_TOKEN", "")).strip()

def _github_raw_url(path_in_repo: str) -> str:
    # acrescenta um parâmetro anti-cache para forçar atualização
    ts = int(time.time())
    return f"https://raw.githubusercontent.com/{REPO_SLUG}/{REPO_BRANCH}/{path_in_repo}?nocache={ts}"

def fmt_brl(x):
    try: return f"R${x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception: return "—"

def to_local_naive_datetime(s: pd.Series, tz: str = LOCAL_TZ) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce", utc=True)     # tudo vira UTC-aware
    s = s.dt.tz_convert(tz).dt.tz_localize(None)         # local + naive
    return s

def pick_col(df: pd.DataFrame, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return default

@st.cache_data(ttl=5.0, show_spinner=False)
def load_csv_smart(path_in_repo: str, local_fallback: str, expected_cols=None, parse_date_cols=None) -> pd.DataFrame:
    """
    1) Tenta GitHub Raw (se REPO_SLUG definido)
    2) Cai para arquivo local no container (clone do repo)
    3) Retorna DataFrame vazio se nada der certo
    """
    # 1) GitHub Raw
    if REPO_SLUG:
        try:
            headers = {}
            if GITHUB_TOKEN:
                headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
            url = _github_raw_url(path_in_repo)
            r = requests.get(url, headers=headers, timeout=15)
            if r.ok and r.text.strip():
                df = pd.read_csv(io.StringIO(r.text))
                if parse_date_cols:
                    for c in parse_date_cols:
                        if c in df.columns:
                            df[c] = pd.to_datetime(df[c], errors="coerce")
                return df
        except Exception:
            pass

    # 2) Arquivo local
    try:
        # tenta o caminho dentro de dashboard_app/...
        if os.path.exists(path_in_repo):
            df = pd.read_csv(path_in_repo)
        # tenta o fallback local explícito
        elif os.path.exists(local_fallback):
            df = pd.read_csv(local_fallback)
        else:
            return pd.DataFrame(columns=expected_cols or [])
        if parse_date_cols:
            for c in parse_date_cols:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame(columns=expected_cols or [])

# ---------------------- Carregamento ----------------------
dfa = load_csv_smart(CSV_A_REPO_PATH, CSV_A_LOCAL)
dfb = load_csv_smart(CSV_B_REPO_PATH, CSV_B_LOCAL, parse_date_cols=["datetime"])

# ---------------------- Normalização A (tolerante a esquemas) ----------------------
if not dfa.empty:
    dt_entry = pick_col(dfa, ["datetime_entry","entry_datetime","entry_time","datetime"])
    symbol   = pick_col(dfa, ["symbol","asset","pair"], "")
    direction= pick_col(dfa, ["direction","dir","signal_dir"], "")
    stake    = pd.to_numeric(pick_col(dfa, ["stake","amount","valor","value"], np.nan), errors="coerce")
    pnl      = pd.to_numeric(pick_col(dfa, ["pnl","profit","resultado"], np.nan), errors="coerce")
    balance  = pd.to_numeric(pick_col(dfa, ["balance_after","balance","saldo"], np.nan), errors="coerce")
    account  = pick_col(dfa, ["account_type","mode","account"], "PRACTICE")
    strategy = pick_col(dfa, ["strategy_name","strategy","estrategia"], "")
    entryref = pick_col(dfa, ["entry_reference","entry_ref","roid","order_id"], "")
    payout   = pd.to_numeric(pick_col(dfa, ["payout","payout_obs"], np.nan), errors="coerce")
    mg_level = pd.to_numeric(pick_col(dfa, ["mg_level","mg","martingale_level"], np.nan), errors="coerce")
    result   = pick_col(dfa, ["result","status","outcome","won"], "")
    if result is not None and (getattr(result, "dtype", None) == bool or set(pd.Series(result).dropna().unique()).issubset({0,1,True,False})):
        result = pd.Series(np.where(pd.Series(result).astype(float)>0.0, "WIN", "LOSS"))
    won = (pd.Series(result).astype(str).str.upper() == "WIN").astype(float)

    if dt_entry is not None:
        dt_entry = to_local_naive_datetime(dt_entry)
    dfa_u = pd.DataFrame({
        "datetime": dt_entry,
        "symbol": symbol,
        "direction": pd.Series(direction).astype(str).str.lower(),
        "stake": stake,
        "pnl": pnl,
        "balance": balance,
        "won": won,
        "account_type": account,
        "strategy": strategy,
        "entry_ref": entryref,
        "payout": payout,
        "mg_level": mg_level,
    }).dropna(subset=["datetime"])
else:
    dfa_u = pd.DataFrame(columns=[
        "datetime","symbol","direction","stake","pnl","balance","won",
        "account_type","strategy","entry_ref","payout","mg_level"
    ])

# ---------------------- Normalização B (legado simples) ----------------------
if not dfb.empty:
    dfb_u = pd.DataFrame({
        "datetime": to_local_naive_datetime(dfb["datetime"]),
        "symbol": dfb.get("asset",""),
        "direction": "",
        "stake": pd.to_numeric(dfb.get("stake"), errors="coerce"),
        "pnl": pd.to_numeric(dfb.get("pnl"), errors="coerce"),
        "balance": np.nan,
        "won": np.where(dfb.get("result","").astype(str).str.upper()=="WIN",1.0,0.0),
        "account_type": dfb.get("mode","PRACTICE"),
        "strategy": dfb.get("strategy",""),
        "entry_ref": dfb.get("roid",""),
        "payout": pd.to_numeric(dfb.get("payout"), errors="coerce"),
        "mg_level": np.nan,
    })
else:
    dfb_u = pd.DataFrame(columns=dfa_u.columns)

# ---------------------- Merge + ordenação ----------------------
df = pd.concat([dfa_u, dfb_u], ignore_index=True)
df = df.dropna(subset=["datetime"]).sort_values("datetime")

st.markdown(f"### {APP_TITLE}")
if df.empty:
    st.info("Nenhuma operação registrada ainda.")
    st.stop()

# ---------------------- Sidebar única / Filtros ----------------------
with st.sidebar:
    st.markdown("## ⚙️ Opções")
    auto = st.toggle("🔄 Auto-atualizar", value=True, help="Recarrega em ~5s", key="k_auto")

    min_day = df["datetime"].dt.date.min()
    max_day = df["datetime"].dt.date.max()
    d_ini, d_fim = st.date_input(
        "📅 Intervalo de datas",
        value=(min_day, max_day),
        min_value=min_day,
        max_value=max_day,
        key="k_date_range",
    )

    st.markdown("**🕒 Janela de horário (local)**")
    colh1, colh2 = st.columns(2)
    h_ini = colh1.time_input("Início", dt.time(0, 0), key="k_time_ini")
    h_fim = colh2.time_input("Fim", dt.time(23, 59), key="k_time_fim")

    ativos = sorted(df["symbol"].dropna().unique().tolist())
    f_ativos = st.multiselect("💹 Ativos", options=ativos, default=ativos, key="k_assets")

    estrategias = sorted([s for s in df["strategy"].dropna().unique().tolist() if s != ""])
    f_estrat = st.multiselect("🧠 Estratégia", options=estrategias, default=estrategias, key="k_strats")

# ---------------------- Aplicação dos filtros ----------------------
d_ini_dt = pd.to_datetime(d_ini).date()
d_fim_dt = pd.to_datetime(d_fim).date()

mask = (df["datetime"].dt.date >= d_ini_dt) & (df["datetime"].dt.date <= d_fim_dt)
df["H"] = df["datetime"].dt.time
mask &= (df["H"] >= h_ini) & (df["H"] <= h_fim)

if f_ativos:
    mask &= df["symbol"].isin(f_ativos)
if f_estrat:
    mask &= df["strategy"].isin(f_estrat)

vf = df.loc[mask].copy()
vf.drop(columns=["H"], inplace=True, errors="ignore")

if vf.empty:
    st.warning("Nenhum resultado com os filtros atuais.")
    if auto:
        time.sleep(5)
        st.rerun()
    st.stop()

# ---------------------- KPIs ----------------------
total = len(vf)
wins = int((vf["pnl"] > 0).sum())
winrate = (wins / total * 100.0) if total else 0.0
pnl_total = float(vf["pnl"].sum())

c1, c2, c3 = st.columns(3)
c1.metric("📦 Operações", f"{total}")
c2.metric("✅ Winrate", f"{winrate:.1f}%")
c3.metric("💰 PnL Total", fmt_brl(pnl_total))

# ---------------------- Curva de Equity -------------------
st.markdown("#### 📈 Curva de Saldo / Equity")
eq_df = vf[["datetime","balance"]].dropna().drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime")
if not eq_df.empty:
    st.line_chart(eq_df.set_index("datetime")["balance"])
else:
    st.info("Sem dados de saldo para plotar a curva de equity.")

# ---------------------- PnL por Dia -----------------------
st.markdown("#### 📊 PnL por Dia")
pnl_day = vf.copy()
pnl_day["Dia"] = pnl_day["datetime"].dt.date
pnl_by_day = pnl_day.groupby("Dia")["pnl"].sum()
st.bar_chart(pnl_by_day)

# ---------------------- Desempenho por Estratégia ---------
st.markdown("#### 🧠 Desempenho por Estratégia")
if vf["strategy"].replace("", np.nan).notna().any():
    rank_strat = vf.replace({"strategy": {"": np.nan}}).dropna(subset=["strategy"]).groupby("strategy").agg(
        Ops=("pnl","count"),
        **{"Winrate %": ("pnl", lambda s: (s > 0).mean() * 100)},
        PnL=("pnl","sum"),
    ).reset_index().sort_values(["PnL","Winrate %","Ops"], ascending=[False, False, False])
    st.dataframe(rank_strat, use_container_width=True)
else:
    st.info("Ainda não há metadados de estratégia no diário.")

# ---------------------- Diário de Operações ----------------
st.markdown("#### 🧾 Diário de Operações (filtrado)")
view = vf[["datetime","symbol","stake","pnl","account_type","strategy","entry_ref"]].copy()
view.columns = ["Data/Hora","Ativo","Stake","PnL","Conta","Estratégia","Referência"]
st.dataframe(view.tail(300), use_container_width=True)

st.caption("Fontes: dashboard_app/log/trader.csv e dashboard_app/data/trade_log.csv (GitHub Raw se configurado).")

# ---------------------- Auto-refresh ----------------------
if st.session_state.get("k_auto", True):
    time.sleep(5)
    st.rerun()
