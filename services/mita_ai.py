from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from data_processor import (
    calcular_estoque,
    listar_precos_consolidados,
    load_metas_local,
    load_registros_caixas,
)
from db import fetch_cache

logger = logging.getLogger(__name__)

DEFAULT_AI_BASE_URL = "https://api.x.ai/v1"
DEFAULT_AI_MODEL = "grok-4-1-fast-reasoning"


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = ""


class MitaChatRequest(BaseModel):
    message: str = Field(default="", min_length=1)
    history: list[ChatHistoryMessage] = Field(default_factory=list)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip().lower()


def _format_currency(value: float | int | None) -> str:
    number = float(value or 0.0)
    formatted = f"{number:,.2f}"
    return f"R$ {formatted.replace(',', 'X').replace('.', ',').replace('X', '.')}"


def _format_number(value: float | int | None, suffix: str = "") -> str:
    number = float(value or 0.0)
    formatted = f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{formatted}{suffix}"


def _serialize_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _safe_float(value) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _serialize_records(records: list[dict], limit: int | None = None) -> list[dict]:
    selected = records[:limit] if limit is not None else records
    serialized: list[dict] = []
    for row in selected:
        serialized.append({key: _serialize_value(value) for key, value in row.items()})
    return serialized


def _summarize_estoque(historico: list[dict]) -> dict:
    movimentos_por_produto: dict[str, dict[str, float]] = {}
    for item in historico:
        produto = str(item.get("produto") or "").strip().upper() or "NAO INFORMADO"
        bucket = movimentos_por_produto.setdefault(
            produto,
            {"entradas_kg": 0.0, "saidas_kg": 0.0, "saldo_kg": 0.0},
        )
        quant = _safe_float(item.get("quant"))
        if str(item.get("tipo") or "").strip().lower() == "entrada":
            bucket["entradas_kg"] += quant
            bucket["saldo_kg"] += quant
        else:
            bucket["saidas_kg"] += quant
            bucket["saldo_kg"] -= quant

    ranking = sorted(
        (
            {
                "produto": produto,
                "entradas_kg": round(dados["entradas_kg"], 2),
                "saidas_kg": round(dados["saidas_kg"], 2),
                "saldo_kg": round(dados["saldo_kg"], 2),
            }
            for produto, dados in movimentos_por_produto.items()
        ),
        key=lambda item: item["saldo_kg"],
    )

    return {
        "ultimas_movimentacoes": _serialize_records(list(reversed(historico[-20:])), limit=20),
        "saldo_por_produto": ranking[:20],
    }


def _summarize_caixas(df_caixas: pd.DataFrame) -> dict:
    if df_caixas.empty:
        return {"resumo_por_loja": [], "ultimos_registros": []}

    numeric_cols = [
        "caixas_benverde",
        "caixas_ccj",
        "ccj_banca",
        "ccj_mercadoria",
        "ccj_retirada",
        "caixas_bananas",
        "total",
    ]
    grouped = (
        df_caixas.groupby("loja", dropna=False)[numeric_cols]
        .sum()
        .reset_index()
        .sort_values("total", ascending=False)
    )
    resumo_por_loja = _serialize_records(grouped.to_dict(orient="records"), limit=20)

    recentes = df_caixas.sort_values(["data", "loja"], ascending=[False, True]).head(20)
    return {
        "resumo_por_loja": resumo_por_loja,
        "ultimos_registros": _serialize_records(recentes.to_dict(orient="records")),
    }


def _summarize_metas(df_metas: pd.DataFrame, pedidos: list[dict]) -> dict:
    metas: list[dict] = []
    pedidos_por_produto: dict[str, float] = {}
    for pedido in pedidos:
        produto = str(pedido.get("produto") or pedido.get("Produto") or "").strip().upper()
        if not produto:
            continue
        pedidos_por_produto[produto] = pedidos_por_produto.get(produto, 0.0) + _safe_float(
            pedido.get("quant") or pedido.get("QUANT")
        )

    if not df_metas.empty:
        for _, row in df_metas.iterrows():
            produto = str(row.get("Produto") or "").strip().upper()
            meta = int(_safe_float(row.get("Meta")))
            pedido_atual = round(pedidos_por_produto.get(produto, 0.0), 2)
            saldo_meta = round(meta - pedido_atual, 2)
            metas.append(
                {
                    "produto": produto,
                    "meta": meta,
                    "pedido_atual": pedido_atual,
                    "faltante_para_meta": saldo_meta,
                    "atingiu_meta": pedido_atual >= meta if meta > 0 else False,
                }
            )

    metas.sort(key=lambda item: item["faltante_para_meta"], reverse=True)
    return {"comparativo_meta": metas[:30]}


def _flatten_cache_pedidos(cache_pedidos: dict) -> list[dict]:
    registros: list[dict] = []
    for arquivo_pdf, items in cache_pedidos.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            registros.append(
                {
                    "arquivo_pdf": arquivo_pdf,
                    "data": item.get("data"),
                    "loja": item.get("loja"),
                    "produto": item.get("produto"),
                    "unidade": item.get("unidade"),
                    "quant": _safe_float(item.get("quant")),
                    "valor_total": _safe_float(item.get("valor_total")),
                    "valor_unit": _safe_float(item.get("valor_unit")),
                }
            )
    return registros


def _summarize_pedidos(pedidos: list[dict]) -> dict:
    if not pedidos:
        return {"registros_recentes": [], "resumo_por_loja": [], "resumo_por_produto": []}

    df = pd.DataFrame(pedidos)
    for col in ["quant", "valor_total", "valor_unit", "QUANT", "VALOR TOTAL", "VALOR UNIT"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "produto" not in df.columns and "Produto" in df.columns:
        df["produto"] = df["Produto"]
    if "loja" not in df.columns and "Loja" in df.columns:
        df["loja"] = df["Loja"]
    if "quant" not in df.columns and "QUANT" in df.columns:
        df["quant"] = df["QUANT"]
    if "valor_total" not in df.columns and "VALOR TOTAL" in df.columns:
        df["valor_total"] = df["VALOR TOTAL"]
    if "valor_unit" not in df.columns and "VALOR UNIT" in df.columns:
        df["valor_unit"] = df["VALOR UNIT"]

    resumo_loja = (
        df.groupby("loja", dropna=False)[["quant", "valor_total"]]
        .sum()
        .reset_index()
        .sort_values("valor_total", ascending=False)
    )
    resumo_produto = (
        df.groupby("produto", dropna=False)[["quant", "valor_total"]]
        .sum()
        .reset_index()
        .sort_values("quant", ascending=False)
    )

    if "data" in df.columns:
        df["data"] = pd.to_datetime(df["data"], errors="coerce")
        recentes = df.sort_values(["data", "produto"], ascending=[False, True]).head(20)
    else:
        recentes = df.head(20)

    return {
        "registros_recentes": _serialize_records(recentes.to_dict(orient="records")),
        "resumo_por_loja": _serialize_records(resumo_loja.to_dict(orient="records"), limit=20),
        "resumo_por_produto": _serialize_records(
            resumo_produto.to_dict(orient="records"), limit=20
        ),
    }


def build_mita_context() -> dict:
    base_dir = Path(__file__).resolve().parent.parent
    pasta_entradas = base_dir / "dados" / "entradas_bananas"
    pasta_saidas = base_dir / "dados" / "saidas_bananas"
    pasta_precos = base_dir / "dados" / "precos"

    saldo, historico = calcular_estoque(
        pasta_entradas=str(pasta_entradas),
        pasta_saidas=str(pasta_saidas),
    )
    precos = listar_precos_consolidados(str(pasta_precos))
    df_caixas = load_registros_caixas()
    df_metas = load_metas_local("")

    pedidos = _flatten_cache_pedidos(fetch_cache("cache_pedidos"))

    context = {
        "gerado_em": datetime.now().isoformat(),
        "estoque": {
            "saldo_atual_kg": round(_safe_float(saldo), 2),
            **_summarize_estoque(historico),
        },
        "caixas": _summarize_caixas(df_caixas),
        "precos": {
            "quantidade_itens": len(precos),
            "itens": _serialize_records(precos, limit=100),
        },
        "metas": _summarize_metas(df_metas, pedidos),
        "pedidos_importados": {
            "quantidade_registros": len(pedidos),
            **_summarize_pedidos(pedidos),
        },
    }
    return context


def buscar_preco_fallback(message: str, precos: list[dict]) -> str | None:
    normalized_message = _normalize_text(message)
    is_price_question = any(
        token in normalized_message
        for token in ("preco", "valor", "custa", "custando", "quanto ta", "quanto esta")
    )
    if not is_price_question or not precos:
        return None

    matches: list[dict] = []
    for item in precos:
        produto = str(item.get("Produto") or "").strip()
        produto_norm = _normalize_text(produto)
        if not produto_norm:
            continue
        if produto_norm in normalized_message or all(
            token in normalized_message for token in produto_norm.split()
        ):
            matches.append(item)

    if not matches:
        return None

    if len(matches) == 1:
        item = matches[0]
        produto = str(item.get("Produto") or "").strip()
        preco = _format_currency(item.get("Preco"))
        return (
            f"O preco atual registrado para {produto} e {preco}. "
            "Estou considerando apenas os dados atuais cadastrados no banco."
        )

    linhas = ["Encontrei mais de um preco relacionado ao que voce perguntou:"]
    for item in matches[:8]:
        linhas.append(
            f"- {item.get('Produto')}: {_format_currency(item.get('Preco'))}"
        )
    linhas.append("Se quiser, posso detalhar um produto especifico.")
    return "\n".join(linhas)


def _create_openai_client():
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError as exc:
        logger.warning("Pacote openai indisponivel: %s", exc)
        return None

    base_url = os.environ.get("AI_BASE_URL", DEFAULT_AI_BASE_URL).strip() or DEFAULT_AI_BASE_URL
    return OpenAI(api_key=api_key, base_url=base_url)


def _build_system_prompt(has_prior_history: bool) -> str:
    intro_rule = (
        "No primeiro turno, voce pode se apresentar brevemente como Mita antes de responder."
        if not has_prior_history
        else "Como a conversa ja esta em andamento, responda direto."
    )
    return (
        "Voce e a Mita, gerente de dados da Benverde.\n"
        "Responda sempre em portugues do Brasil.\n"
        "Use exclusivamente os dados fornecidos no contexto.\n"
        "Nunca invente numeros, datas, produtos ou conclusoes sem base.\n"
        "Se faltar informacao, diga claramente que nao ha registro suficiente.\n"
        "Seja objetiva, analitica e util para a operacao.\n"
        "Quando citar valores monetarios, use formato BRL como R$ 1.234,56.\n"
        "Quando citar quantidades, deixe a unidade explicita.\n"
        f"{intro_rule}"
    )


def _sanitize_history(history: list[ChatHistoryMessage] | list[dict]) -> list[dict]:
    sanitized: list[dict] = []
    for item in history or []:
        role = getattr(item, "role", None) if not isinstance(item, dict) else item.get("role")
        content = (
            getattr(item, "content", None) if not isinstance(item, dict) else item.get("content")
        )
        if role not in {"user", "assistant"}:
            continue
        text = str(content or "").strip()
        if not text:
            continue
        sanitized.append({"role": role, "content": text})
    return sanitized


def _generate_answer_with_ai(message: str, history: list[dict], context: dict) -> str:
    client = _create_openai_client()
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="IA indisponivel no backend: configure XAI_API_KEY e a dependencia openai.",
        )

    model = os.environ.get("AI_MODEL", DEFAULT_AI_MODEL).strip() or DEFAULT_AI_MODEL
    messages = [
        {"role": "system", "content": _build_system_prompt(has_prior_history=bool(history))},
        {
            "role": "system",
            "content": "Contexto operacional atual em JSON:\n" + json.dumps(
                context, ensure_ascii=False, default=str
            ),
        },
        *history,
        {"role": "user", "content": message},
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
    except Exception as exc:
        logger.exception("Falha ao consultar a IA da Mita")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A IA da Mita esta indisponivel no momento.",
        ) from exc

    answer = ""
    if completion.choices:
        answer = (completion.choices[0].message.content or "").strip()
    if not answer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="A IA da Mita nao retornou resposta utilizavel.",
        )
    return answer


def chat_with_mita(message: str, history: list[ChatHistoryMessage] | list[dict]) -> dict:
    cleaned_message = str(message or "").strip()
    if not cleaned_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O campo 'message' e obrigatorio.",
        )

    sanitized_history = _sanitize_history(history)

    try:
        context = build_mita_context()
        fallback_answer = buscar_preco_fallback(cleaned_message, context["precos"]["itens"])
        answer = fallback_answer or _generate_answer_with_ai(
            message=cleaned_message,
            history=sanitized_history,
            context=context,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erro inesperado no chat da Mita")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno inesperado ao consultar a Mita.",
        ) from exc

    updated_history = [
        *sanitized_history,
        {"role": "user", "content": cleaned_message},
        {"role": "assistant", "content": answer},
    ]
    return {
        "answer": answer,
        "history": updated_history,
        "context_summary": {
            "saldo_estoque_kg": _format_number(context["estoque"]["saldo_atual_kg"], " kg"),
            "itens_com_preco": context["precos"]["quantidade_itens"],
            "registros_pedidos": context["pedidos_importados"]["quantidade_registros"],
        },
    }
