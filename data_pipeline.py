# data_pipeline.py
from __future__ import annotations
import re
import json
import unicodedata
from typing import Dict, Any, Optional
import pandas as pd
import datetime

# ---------- utilidades ----------
def _rm_accents(s: str) -> str:
    if not isinstance(s, str):
        return s
    return ''.join(
        c for c in unicodedata.normalize('NFKD', s)
        if not unicodedata.combining(c)
    )

def _norm_key(s: str) -> str:
    """Chave canônica de produto: sem acento, maiúscula, sem espaços duplicados."""
    if s is None:
        return s
    s2 = _rm_accents(s).strip().upper()
    s2 = re.sub(r'\s+', ' ', s2)
    return s2

def _parse_preco_raw(valor_raw: Optional[str]) -> Optional[float]:
    """Aceita formatos: 'R$ 4,89', '4.89', 'R 4,89', 4.89 -> retorna float (ex: 4.89)"""
    if valor_raw is None:
        return None
    v = str(valor_raw).strip()
    if v == "" or v.lower() in ("nan", "indisponivel", "nao encontrado", "não encontrado"):
        return None
    # remove símbolos de moeda e espaços
    v = v.replace("R$", "").replace("r$", "").replace("R", "").replace("$", "").strip()
    # limpa texto restante (mantém apenas dígitos, ponto e vírgula)
    v = re.sub(r"[^\d\.,]", "", v)

    if "," in v and "." in v:
        # Ex.: 1.234,56 -> 1234.56
        v = v.replace(".", "").replace(",", ".")
    elif "," in v:
        # Ex.: 4,89 -> 4.89
        v = v.replace(",", ".")
    elif "." in v and v.count(".") > 1:
        # Ex.: 1.234.567 -> 1234567
        v = v.replace(".", "")

    try:
        return float(v) if v != "" else None
    except Exception:
        return None

def _ultima_data_do_dict(precos_dict: Dict[str, Any]) -> str:
    """Retorna a key com a última data no formato dd-mm-YYYY (assume chaves nesse estilo)."""
    keys = list(precos_dict.keys())
    # tentar parse seguro
    parsed = []
    for k in keys:
        try:
            d = datetime.datetime.strptime(k, "%d-%m-%Y")
            parsed.append((d, k))
        except Exception:
            # tenta outras formas ou ignora
            try:
                d = datetime.datetime.strptime(k, "%Y-%m-%d")
                parsed.append((d, k))
            except Exception:
                pass
    if not parsed:
        # fallback: ordena alfabeticamente
        return sorted(keys)[-1]
    parsed.sort()
    return parsed[-1][1]

# ---------- estruturar precos ----------
def estruturar_precos(df_precos: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Espera DataFrame com colunas como:
    'Produto Buscado', 'Preço (Semar)', 'Status (Semar)', 'Preço (Rossi)', 'Status (Rossi)', ...
    Retorna: { "BANANA NANICA": {"Semar": 2.99, "Rossi": 3.98}, ... }
    """
    resultado: Dict[str, Dict[str, float]] = {}
    if df_precos is None or df_precos.empty:
        return resultado

    store_patterns = []
    # descobrir dinamicamente nomes de lojas pelas colunas
    for col in df_precos.columns:
        m = re.match(r"Preço \((.+)\)", col)
        if m:
            store_patterns.append(m.group(1))

    for _, row in df_precos.iterrows():
        prod_bruto = row.get("Produto Buscado") or row.get("Produto") or row.get("Produto Buscado (1)") 
        if not prod_bruto:
            continue
        produto_key = _norm_key(prod_bruto)
        precos_por_loja: Dict[str, float] = {}
        for loja in store_patterns:
            preco_col = f"Preço ({loja})"
            status_col = f"Status ({loja})"
            raw_preco = row.get(preco_col, None)
            status = str(row.get(status_col, "")).strip().upper()
            preco = _parse_preco_raw(raw_preco)
            if preco is not None and (status == "" or status == "OK"):
                precos_por_loja[loja] = preco
        if precos_por_loja:
            resultado[produto_key] = precos_por_loja
    return resultado

# ---------- estruturar metas ----------
def estruturar_metas(df_metas: pd.DataFrame) -> Dict[str, int]:
    """
    Espera DataFrame com colunas 'Produto' e 'Meta' (metas em KG ou UN conforme cadastro).
    Retorna: { "TOMATE": 684560, ... }
    """
    resultado: Dict[str, int] = {}
    if df_metas is None or df_metas.empty:
        return resultado
    for _, row in df_metas.iterrows():
        prod = row.get("Produto") or row.get("Produto ")
        if not prod:
            continue
        k = _norm_key(prod)
        try:
            m = int(row.get("Meta", 0))
        except Exception:
            try:
                m = int(float(str(row.get("Meta", 0)).replace(",", ".")))
            except Exception:
                m = 0
        resultado[k] = m
    return resultado

# ---------- estruturar progresso ----------
def estruturar_progresso(df_prog: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Espera DataFrame com colunas: Produtos, meta, pedido, Progresso (número percentual), status da meta
    Retorna: {
        "TOMATE": {"meta": 684560, "pedido": 6164.0, "progresso_pct": 0.9}
    }
    """
    resultado: Dict[str, Dict[str, Any]] = {}
    if df_prog is None or df_prog.empty:
        return resultado
    for _, row in df_prog.iterrows():
        prod = row.get("Produtos") or row.get("Produtos ")
        if not prod:
            continue
        k = _norm_key(prod)
        try:
            meta = float(row.get("meta", 0))
        except Exception:
            meta = 0.0
        try:
            pedido = float(row.get("pedido", 0))
        except Exception:
            pedido = 0.0
        # Progresso no seu dump já vem como número (ex: 0.9 ou 156.0) — manter como float %
        try:
            prog = float(row.get("Progresso", 0.0))
        except Exception:
            prog = 0.0
        resultado[k] = {
            "meta": meta,
            "pedido": pedido,
            "progresso_pct": prog
        }
    return resultado

# ---------- ponto único de montagem ----------
def montar_dados_para_llm(precos_dict: Dict[str, Any],
                          metas_df: pd.DataFrame,
                          progresso_df: pd.DataFrame,
                          saldo_estoque: float,
                          historico_estoque: list) -> Dict[str, Any]:
    """
    Retorna JSON limpo pronto para injetar no prompt do LLM.
    """
    ultima_key = _ultima_data_do_dict(precos_dict) if precos_dict else None
    df_precos = precos_dict.get(ultima_key) if ultima_key else None

    dados = {
        "data_referencia": ultima_key,
        "precos": estruturar_precos(df_precos) if df_precos is not None else {},
        "metas": estruturar_metas(metas_df),
        "progresso": estruturar_progresso(progresso_df),
        "estoque_banana_kg": float(saldo_estoque or 0.0),
        # opcional: enviar últimos N registros do historico em forma reduzida
        "historico_estoque_ultimos": [
            {
                "data": h.get("data").strftime("%Y-%m-%d") if isinstance(h.get("data"), (datetime.datetime, datetime.date)) else str(h.get("data")),
                "produto": _norm_key(h.get("produto")),
                "quant_kg": float(h.get("quant", 0.0)),
                "tipo": h.get("tipo")
            }
            for h in (historico_estoque or [])[:5]
        ]
    }
    return dados
