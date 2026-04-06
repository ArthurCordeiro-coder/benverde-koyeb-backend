"""
data_processor.py
Módulo de processamento de dados para sistema de gerenciamento de empresa hortifrúti.
Lida com CSVs de preços, JSON de metas/vendas e PDFs de notas fiscais (DANFEs e Pedidos Semar).

Otimizações para E5-2620 v3 (6 cores / 12 threads):
    - Extração via extract_text() + regex (5-10x mais rápida que extract_tables())
    - ProcessPoolExecutor com workers reais (usa todos os núcleos físicos)
    - Cache incremental com salvamento a cada lote (resiliente a interrupções)
    - Regex compilados em nível de módulo (sem recompilação a cada chamada)
"""

import os
import glob
import re
import json
import logging
import shutil
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta
from typing import Callable, Optional

import pandas as pd
import pdfplumber

from db import (
    delete_movimentacao,
    fetch_cache,
    fetch_caixas,
    fetch_precos_rows,
    get_import_job,
    fetch_movimentacoes,
    insert_caixa,
    insert_movimentacoes,
    load_metas,
    replace_metas,
    replace_cache_pedidos,
    update_import_job,
    upsert_cache,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

IMPORT_JOB_LOG_LIMIT = 25
IMPORT_JOB_HEARTBEAT_SECONDS = 5

# ---------------------------------------------------------------------------
# Regex compilados em nível de módulo (performance)
# ---------------------------------------------------------------------------

# Produto em linha DANFE: CÓD DESCRIÇÃO NCM(8d) CST(3d) CFOP(4d) UNID QUANT V.UNIT V.TOTAL
_RE_DANFE_LINHA = re.compile(
    r'^\s*\d+\s+'
    r'([A-ZÁÀÃÂÉÊÍÓÔÕÚÇ\w][^\n\d]{2,60}?)\s+'
    r'\d{8}\s+\d{3}\s+\d{4}\s+'
    r'(KG|UN|CX|FD|PCT|SC|BAG)\s+'
    r'([\d.,]+)\s+'
    r'([\d.,]+)\s+'
    r'([\d.,]+)',
    re.IGNORECASE | re.MULTILINE,
)

# Banana em DANFE
_RE_BANANA = re.compile(r'BANANA[\s\-]', re.IGNORECASE)

# Unidade + quantidade no campo NCM mesclado
_RE_UN_QT_NCM = re.compile(r'\b(KG|UN|CX|FD|PCT|SC|BAG)\s+([\d.,]+)', re.IGNORECASE)

# Linha de código de barras
_RE_COD_BARRAS = re.compile(r'^c[oó]d\.?\s*(?:de\s*)?barras?[\s:]*[\d\s]+$', re.IGNORECASE)

# Só dígitos/espaços (NCM, EAN solto)
_RE_SO_DIGITOS = re.compile(r'^[\d\s]{8,}$')

# NCM válido na célula (sub-linha real de produto)
_RE_TEM_NCM = re.compile(r'\b(KG|UN|CX|FD|PCT|SC|BAG)\s+[\d.,]+', re.IGNORECASE)

# Lixo nas células de descrição
_RE_LIXO_DESC = re.compile(
    r'^(c[oó]d\.?\s*(?:de\s*)?barras?[\s:]*[\d\s]+|[\d]{6,}|\s*)$',
    re.IGNORECASE,
)

# Produto Semar
_RE_PROD_SEMAR  = re.compile(r'^([A-ZÁÀÃÂÉÊÍÓÔÕÚÇ][A-ZÁÀÃÂÉÊÍÓÔÕÚÇ\s]+?)\s+kg\b', re.IGNORECASE)
_RE_DATA_SEMAR  = re.compile(r'Data de emiss[aã]o[:\s]+(\d{2}/\d{2}/\d{4})', re.IGNORECASE)
_RE_CUSTO_SEMAR = re.compile(r'(\d+[.,]\d+)')
_RE_LOJA_SEMAR  = re.compile(r'LOJA\s+(\d+)', re.IGNORECASE)
_RE_QUANT_SEMAR = re.compile(r'^[\d.,]+$')

# Peso por caixa embutido no nome: "BETERRABA KG CX 20" → 20 kg/cx
_RE_KG_CX = re.compile(r'\bKG\s+CX\s+([\d]+(?:[.,]\d+)?)\b', re.IGNORECASE)

# Meses PT
_MESES_PT = {
    "jan": 1,  "janeiro":   1,
    "fev": 2,  "fevereiro": 2,
    "mar": 3,  "marco":     3,  "março": 3,
    "abr": 4,  "abril":     4,
    "mai": 5,  "maio":      5,
    "jun": 6,  "junho":     6,
    "jul": 7,  "julho":     7,
    "ago": 8,  "agosto":    8,
    "set": 9,  "setembro":  9,
    "out": 10, "outubro":   10,
    "nov": 11, "novembro":  11,
    "dez": 12, "dezembro":  12,
}

_LOJAS_ESPECIAIS = {"libra": "Frutas/Legumes", "suzano": "Frutas/Legumes"}

_RE_LOJA_PREFIXO = re.compile(r'(?:loja|lj)\s*(\d{1,2})', re.IGNORECASE)
_RE_LOJA_NUMERO  = re.compile(r'^\s*(\d{1,2})\b')
_RE_MES_COLADO   = re.compile(r'(\d)([a-záàãâéêíóôõúç])', re.IGNORECASE)
_RE_MES_COLADO2  = re.compile(r'([a-záàãâéêíóôõúç])(\d)', re.IGNORECASE)
_PAT_MES_TEXTO   = re.compile(
    r'^(\d{1,2})\s+(' + '|'.join(_MESES_PT.keys()) + r')\b(.*)',
    re.IGNORECASE,
)
# ---------------------------------------------------------------------------
# Helpers numéricos
# ---------------------------------------------------------------------------

def _parse_br(val) -> float:
    """Converte string BR (vírgula decimal, ponto milhar) para float."""
    s = str(val or "").strip().replace(" ", "")
    if not s or s.lower() in ("none", "nan", "-", ""):
        return 0.0
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    else:
        partes = s.split(".")
        if len(partes) == 2 and len(partes[1]) == 3:
            s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _parse_numero(valor) -> Optional[float]:
    """Converte valor para float; retorna None se falhar."""
    if valor is None:
        return None
    try:
        s = str(valor).strip().replace(" ", "")
        if not s or s.lower() in ("none", "nan", "-", ""):
            return None
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        return float(s)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Conversão CX → KG para produtos com peso/caixa no nome
# ---------------------------------------------------------------------------

def _resolver_cx_para_kg(produto: str, quant: float, unidade: str):
    """Converte caixas → KG quando o nome indica 'KG CX N'.

    Ex: "BETERRABA KG CX 20", unidade="CX", quant=5
        → ("BETERRABA", 100.0, "KG")

    Produtos com outras unidades ou sem o padrão no nome não são alterados.
    """
    if unidade.upper() != "CX":
        return produto, quant, unidade
    m = _RE_KG_CX.search(produto)
    if not m:
        return produto, quant, unidade
    kg_por_cx = _parse_br(m.group(1))
    if kg_por_cx <= 0:
        return produto, quant, unidade
    nome_limpo = _RE_KG_CX.sub("", produto).strip().strip("-").strip()
    return nome_limpo, round(quant * kg_por_cx, 3), "KG"


# ---------------------------------------------------------------------------
# Parse de data em nome de arquivo
# ---------------------------------------------------------------------------

def _extrair_data_saida_pdf(caminho_pdf: str) -> Optional[datetime]:
    """Lê a DATA DA SAÍDA diretamente do texto de um DANFE (NF-e).

    Procura pelo rótulo 'DATA DA SAÍDA' e captura o primeiro DD/MM/AAAA após ele.
    Retorna None se não encontrar ou falhar.
    """
    _RE = re.compile(
        r'DATA\s+DA\s+SA[IÍ]DA.{0,150}?(\d{2}/\d{2}/\d{4})',
        re.IGNORECASE | re.DOTALL,
    )
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            for pagina in pdf.pages:
                texto = pagina.extract_text(x_tolerance=3, y_tolerance=3) or ""
                m = _RE.search(texto)
                if m:
                    try:
                        return datetime.strptime(m.group(1), "%d/%m/%Y")
                    except ValueError:
                        pass
    except Exception as exc:
        logger.debug("Falha ao extrair DATA DA SAÍDA de '%s': %s", caminho_pdf, exc)
    return None


def _texto_primeira_pagina_pdf(caminho_pdf: str) -> str:
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            if not pdf.pages:
                return ""
            return pdf.pages[0].extract_text(x_tolerance=3, y_tolerance=3) or ""
    except Exception as exc:
        logger.debug("Falha ao ler a primeira pagina de '%s': %s", caminho_pdf, exc)
        return ""


def extrair_metadados_pdf(texto_pagina: str):
    data_emissao = datetime.now()
    loja = "Loja ?"
    municipio = "Desconhecido"

    if not texto_pagina:
        return data_emissao, loja, municipio

    match_data = re.search(r"(\d{2}/\d{2}/\d{4})", texto_pagina)
    if match_data:
        try:
            data_emissao = datetime.strptime(match_data.group(1), "%d/%m/%Y")
        except ValueError:
            pass

    match_loja = re.search(r"(?:LJ|LOJA)\s*(\d{1,2})", texto_pagina, re.IGNORECASE)
    if match_loja:
        try:
            numero_loja = int(match_loja.group(1))
            loja = f"Loja {numero_loja:02d}"
        except ValueError:
            pass

    match_municipio = re.search(
        r"MUNIC[IÍ]PIO[\r\n]+([A-ZÀ-Ú\s]+)",
        texto_pagina,
        re.IGNORECASE,
    )
    if match_municipio:
        municipio_cru = match_municipio.group(1).strip().split("\n")[0].strip()
        if municipio_cru:
            municipio = municipio_cru

    return data_emissao, loja, municipio


def _extrair_metadados_pdf(caminho_pdf: str) -> dict:
    texto_bruto = _texto_primeira_pagina_pdf(caminho_pdf)
    data_emissao, loja, municipio = extrair_metadados_pdf(texto_bruto)
    return {
        "data": data_emissao,
        "loja": loja,
        "localizacao": municipio,
    }


def parse_data_arquivo(nome_arq: str) -> Optional[datetime]:
    """Extrai data do nome de arquivo (padrões DD_MM ou DDMM)."""
    basename = os.path.splitext(os.path.basename(nome_arq))[0]
    ano = datetime.now().year

    m = re.search(r'(\d{2})_(\d{2})', basename)
    if m:
        dia, mes = int(m.group(1)), int(m.group(2))
        try:
            dt = datetime(ano, mes, dia)
            if dt > datetime.now() + timedelta(days=7):
                dt = datetime(ano - 1, mes, dia)
            return dt
        except ValueError:
            return None

    m = re.search(r'^(\d{2})(\d{2})', basename)
    if m:
        dia, mes = int(m.group(1)), int(m.group(2))
        try:
            dt = datetime(ano, mes, dia)
            if dt > datetime.now() + timedelta(days=7):
                dt = datetime(ano - 1, mes, dia)
            return dt
        except ValueError:
            return None

    return None


# ---------------------------------------------------------------------------
# Cache incremental
# ---------------------------------------------------------------------------

def _cache_table_for_path(caminho_cache: str, fallback: str) -> str:
    if not caminho_cache:
        return fallback
    path = caminho_cache.lower()
    if "cache_estoque" in path:
        return "cache_estoque"
    if "cache_pedidos" in path or "cache_semar" in path:
        return "cache_pedidos"
    return fallback


def _carregar_cache(caminho_cache: str, fallback: str) -> dict:
    table = _cache_table_for_path(caminho_cache, fallback)
    try:
        return fetch_cache(table)
    except Exception as exc:
        logger.warning("Falha ao carregar cache '%s': %s", caminho_cache, exc)
        return {}


def _salvar_cache(cache: dict, caminho_cache: str, fallback: str) -> None:
    if not cache:
        return
    table = _cache_table_for_path(caminho_cache, fallback)
    try:
        upsert_cache(table, cache)
    except Exception as exc:
        logger.error("Falha ao salvar cache '%s': %s", caminho_cache, exc)


# ---------------------------------------------------------------------------
# Parse de nome de arquivo NF-e (todos os formatos)
# ---------------------------------------------------------------------------

def _identificar_loja(texto_restante: str) -> str:
    t = texto_restante.strip().lstrip("_-,. ").lower()
    for palavra, nome_loja in _LOJAS_ESPECIAIS.items():
        if palavra in t:
            return nome_loja
    m = _RE_LOJA_PREFIXO.search(t)
    if m:
        return f"Loja {int(m.group(1)):02d}"
    m = _RE_LOJA_NUMERO.search(t)
    if m:
        return f"Loja {int(m.group(1)):02d}"
    return "Loja ?"


def _montar_data(dia: int, mes: int, nome_arq: str = "") -> Optional[datetime]:
    ano = datetime.now().year
    try:
        dt = datetime(ano, mes, dia)
        if dt > datetime.now() + timedelta(days=7):
            dt = datetime(ano - 1, mes, dia)
        return dt
    except ValueError:
        logger.error("Data inválida em '%s': dia=%d mes=%d", nome_arq, dia, mes)
        return None


def _parse_nome_arquivo_nfe(nome_arq: str) -> tuple:
    """Extrai (datetime, loja_str) de nomes de arquivo NF-e com formatos variados."""
    basename = os.path.splitext(os.path.basename(nome_arq))[0].strip()

    # Remove pontuacao inicial: ",2301 libra" -> "2301 libra"
    basename = re.sub(r"^[,.\s]+", "", basename).strip()
    # Virgula entre letras: "j,an" -> "jan"
    basename = re.sub(r"([a-z]),([a-z])", r"\1\2", basename, flags=re.IGNORECASE)
    # Demais virgulas -> espaco
    b_norm  = re.sub(r'[,]+', ' ', basename).strip()
    b_norm  = re.sub(r'\s+', ' ', b_norm)

    # 0) Mes por extenso / abreviado
    b_sep = _RE_MES_COLADO.sub(r'\1 \2', b_norm.lower())
    b_sep = _RE_MES_COLADO2.sub(r'\1 \2', b_sep)
    b_sep = re.sub(r'\s+', ' ', b_sep).strip()
    m = _PAT_MES_TEXTO.match(b_sep)
    if m:
        dia  = int(m.group(1))
        mes  = _MESES_PT[m.group(2).lower()]
        loja = _identificar_loja(m.group(3))
        return _montar_data(dia, mes, nome_arq), loja

    # 1) DD.MM
    m = re.match(r'^(\d{2})\.(\d{2})(.*)', basename)
    if m:
        dia, mes = int(m.group(1)), int(m.group(2))
        return _montar_data(dia, mes, nome_arq), _identificar_loja(m.group(3))

    # 2) DD MM (espaço, cobre DD,MM após normalização)
    m = re.match(r'^(\d{1,2})\s+(\d{2})\s*(.*)', b_norm)
    if m:
        dia, mes = int(m.group(1)), int(m.group(2))
        return _montar_data(dia, mes, nome_arq), _identificar_loja(m.group(3))

    # 3) DDMMLL (6 dígitos colados)
    m = re.match(r'^(\d{2})(\d{2})(\d{2})(?:\D|$)', basename)
    if m:
        dia, mes = int(m.group(1)), int(m.group(2))
        return _montar_data(dia, mes, nome_arq), f"Loja {int(m.group(3)):02d}"

    # 4) DDMM + resto
    m = re.match(r'^(\d{2})(\d{2})(.*)', basename)
    if m:
        dia, mes = int(m.group(1)), int(m.group(2))
        return _montar_data(dia, mes, nome_arq), _identificar_loja(m.group(3))

    logger.warning("Nome de arquivo NF-e fora de qualquer padrão reconhecido: '%s'", nome_arq)
    return None, "Loja ?"


# ---------------------------------------------------------------------------
# Índice de coluna
# ---------------------------------------------------------------------------

def _indice_coluna(cabecalho: list, candidatos: list) -> Optional[int]:
    for candidato in candidatos:
        for i, col in enumerate(cabecalho):
            if candidato in col:
                return i
    return None


# ---------------------------------------------------------------------------
# Extrator DANFE: texto+regex (primário) + tabelas (fallback)
# ---------------------------------------------------------------------------

def _extrair_produtos_texto(texto: str) -> list:
    """Extrai produtos de texto bruto via regex — método primário (rápido)."""
    registros = []
    for m in _RE_DANFE_LINHA.finditer(texto):
        desc    = m.group(1).strip().upper()
        unidade = m.group(2).upper()
        quant   = _parse_br(m.group(3))
        vunit   = _parse_br(m.group(4))
        vtotal  = _parse_br(m.group(5))
        if quant <= 0 or len(desc) < 3:
            continue
        if _RE_SO_DIGITOS.match(desc):
            continue
        desc, quant, unidade = _resolver_cx_para_kg(desc, quant, unidade)
        registros.append({
            "produto":     desc,
            "quant":       quant,
            "unidade":     unidade,
            "valor_unit":  vunit,
            "valor_total": vtotal,
        })
    return registros


def _sub_linhas(celula) -> list:
    return [p.strip() for p in str(celula or "").split("\n") if p.strip()]


def _extrair_produtos_tabela(tabela: list) -> list:
    """Extrai produtos de uma tabela pdfplumber — método fallback."""
    if not tabela or len(tabela) < 2:
        return []

    cab = [str(c or "").strip().upper() for c in tabela[0]]
    idx_desc  = _indice_coluna(cab, ["DESCRIÇÃO DO PRODUTO", "DESCRICAO DO PRODUTO",
                                      "DESCRIÇÃO", "DESCRICAO", "PRODUTO"])
    idx_ncm   = _indice_coluna(cab, ["NCM", "NCM/SH", "CST", "CFOP", "UNID"])
    idx_vals  = _indice_coluna(cab, ["VALOR UNIT", "VL UNIT", "VALOR UNITÁRIO", "V.UNIT", "VL.UNIT"])
    idx_total = _indice_coluna(cab, ["VALOR TOTAL", "VL TOTAL", "TOTAL"])

    if idx_desc is None:
        return []

    registros = []
    for linha in tabela[1:]:
        if not linha or len(linha) <= idx_desc:
            continue
        celula_desc = str(linha[idx_desc] or "").strip()
        if not celula_desc:
            continue
        if _RE_COD_BARRAS.match(celula_desc):
            continue
        if _RE_SO_DIGITOS.match(celula_desc):
            continue

        sub_desc  = _sub_linhas(celula_desc)
        sub_ncm   = _sub_linhas(linha[idx_ncm])   if idx_ncm   is not None and idx_ncm   < len(linha) else []
        sub_vals  = _sub_linhas(linha[idx_vals])   if idx_vals  is not None and idx_vals  < len(linha) else []
        sub_total = _sub_linhas(linha[idx_total])  if idx_total is not None and idx_total < len(linha) else []

        sub_linhas_reais = sum(1 for s in sub_ncm if _RE_TEM_NCM.search(s)) if sub_ncm else 0
        quebra_de_texto  = sub_linhas_reais <= 1 and len(sub_desc) > 1

        if quebra_de_texto:
            nomes_validos = [s.strip() for s in sub_desc if s.strip() and not _RE_LIXO_DESC.match(s.strip())]
            sub_desc  = [nomes_validos[0]] if nomes_validos else [sub_desc[0]]
            sub_ncm   = sub_ncm[:1]   if sub_ncm   else []
            sub_vals  = sub_vals[:1]  if sub_vals  else []
            sub_total = sub_total[:1] if sub_total else []

        for i, desc in enumerate(sub_desc):
            desc_up = desc.strip().upper()
            if not desc_up or len(desc_up) < 3:
                continue
            if _RE_SO_DIGITOS.match(desc_up):
                continue
            if _RE_COD_BARRAS.match(desc_up):
                continue

            txt_ncm  = sub_ncm[i]   if i < len(sub_ncm)   else (sub_ncm[-1]  if sub_ncm  else "")
            txt_vals = sub_vals[i]  if i < len(sub_vals)  else (sub_vals[-1] if sub_vals else "")
            txt_tot  = sub_total[i] if i < len(sub_total) else (sub_total[-1] if sub_total else "")

            m_ncm   = _RE_UN_QT_NCM.search(txt_ncm)
            unidade = m_ncm.group(1).upper() if m_ncm else "UN"
            quant   = _parse_br(m_ncm.group(2)) if m_ncm else 0.0

            nums_v = re.findall(r'[\d]+(?:[.,][\d]+)*', txt_vals)
            vunit  = _parse_br(nums_v[0]) if len(nums_v) > 0 else 0.0
            vtotal = _parse_br(nums_v[1]) if len(nums_v) > 1 else 0.0

            if txt_tot:
                nums_t = re.findall(r'[\d]+(?:[.,][\d]+)*', txt_tot)
                if nums_t:
                    vt2 = _parse_br(nums_t[0])
                    if vt2 > 0:
                        vtotal = vt2

            if quant <= 0:
                continue

            desc_up, quant, unidade = _resolver_cx_para_kg(desc_up, quant, unidade)
            registros.append({
                "produto":     desc_up,
                "quant":       quant,
                "unidade":     unidade,
                "valor_unit":  vunit,
                "valor_total": vtotal,
            })
    return registros


def _extrair_todos_produtos_pdf(caminho_pdf: str) -> list:
    """Extrai todos os produtos de um DANFE.

    Estratégia:
        1. extract_text() + regex (rápido, método primário)
        2. extract_tables() como fallback
    """
    arquivo_nome = os.path.basename(caminho_pdf)
    registros    = []

    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            for num_pag, pagina in enumerate(pdf.pages, start=1):
                # Método 1: texto + regex
                try:
                    texto = pagina.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    prods = _extrair_produtos_texto(texto)
                    if prods:
                        registros.extend(prods)
                        continue
                except Exception as e:
                    logger.debug("[PDF] Texto falhou pág %d '%s': %s", num_pag, arquivo_nome, e)

                # Método 2: tabelas fallback
                try:
                    for tabela in (pagina.extract_tables() or []):
                        registros.extend(_extrair_produtos_tabela(tabela))
                except Exception as e:
                    logger.warning("[PDF] Tabela falhou pág %d '%s': %s", num_pag, arquivo_nome, e)

    except Exception as e:
        logger.error("[PDF] Falha ao abrir '%s': %s", arquivo_nome, e)

    # Deduplica
    vistos: set = set()
    unicos = []
    for r in registros:
        chave = (r["produto"], r["quant"], r["valor_total"])
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(r)

    logger.info("[PDF] '%s' → %d produto(s).", arquivo_nome, len(unicos))
    return unicos


# ---------------------------------------------------------------------------
# Extrator Pedido de Compra Semar
# ---------------------------------------------------------------------------

def _extrair_pedido_semar(caminho_pdf: str) -> list:
    """Extrai produtos e quantidades por loja de um Pedido de Compra Semar.

    Delega para extrair_pedido_semar() e converte resultado para lista de dicts
    com chaves minúsculas (interface esperada por _processar_pdf_worker).
    """
    df = extrair_pedido_semar(caminho_pdf)
    if df.empty:
        return []
    return [
        {
            "produto":     row["Produto"],
            "loja":        row["Loja"],
            "quant":       row["QUANT"],
            "valor_unit":  row["VALOR UNIT"],
            "valor_total": row["VALOR TOTAL"],
            "data":        row["Data"],
        }
        for _, row in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Workers de nível de módulo (obrigatório para ProcessPoolExecutor no Windows)
# ---------------------------------------------------------------------------

def _worker_pedido(caminho_pdf: str) -> tuple:
    """Worker para load_pedidos_pdfs."""
    nome_arq = os.path.basename(caminho_pdf)
    metadados = _extrair_metadados_pdf(caminho_pdf)
    dt = metadados.get("data")
    loja = metadados.get("loja") or "Loja ?"
    localizacao = metadados.get("localizacao") or ""
    produtos = _extrair_todos_produtos_pdf(caminho_pdf)
    return (nome_arq, dt, loja, localizacao, produtos)


def _processar_pdf_worker(args: tuple) -> list:
    """Worker para calcular_estoque: detecta DANFE ou Pedido Semar."""
    caminho_pdf, tipo = args
    historico_local   = []
    dt = _extrair_data_saida_pdf(caminho_pdf)
    if dt is None:
        dt = parse_data_arquivo(caminho_pdf)

    eh_semar = False
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            if "pedido de compra" in (pdf.pages[0].extract_text() or "").lower():
                eh_semar = True
    except Exception:
        pass

    if eh_semar:
        for reg in _extrair_pedido_semar(caminho_pdf):
            historico_local.append({
                "data":        reg.get("data") or dt,
                "tipo":        tipo,
                "produto":     reg["produto"],
                "quant":       reg["quant"],
                "unidade":     "KG",
                "valor_unit":  reg["valor_unit"],
                "valor_total": reg["valor_total"],
                "arquivo":     os.path.basename(caminho_pdf),
            })
    else:
        for reg in _extrair_bananas_pdf(caminho_pdf):
            historico_local.append({
                "data":        dt,
                "tipo":        tipo,
                "produto":     reg["produto"],
                "quant":       reg["quant"],
                "unidade":     reg.get("unidade", "KG"),
                "valor_unit":  reg.get("valor_unit", 0.0),
                "valor_total": reg.get("valor_total", 0.0),
                "arquivo":     os.path.basename(caminho_pdf),
            })

    return historico_local


# ---------------------------------------------------------------------------
# Extrator de bananas
# ---------------------------------------------------------------------------

def _extrair_bananas_pdf(caminho_pdf: str) -> list:
    """Extrai apenas linhas de BANANA de um DANFE."""
    arquivo_nome = os.path.basename(caminho_pdf)
    registros    = []

    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            for pagina in pdf.pages:
                try:
                    texto = pagina.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    prods = _extrair_produtos_texto(texto)
                    bananas = [p for p in prods if _RE_BANANA.search(p["produto"])]
                    if bananas:
                        registros.extend(bananas)
                        continue
                except Exception:
                    pass
                try:
                    for tabela in (pagina.extract_tables() or []):
                        todos = _extrair_produtos_tabela(tabela)
                        registros.extend(p for p in todos if _RE_BANANA.search(p["produto"]))
                except Exception:
                    pass
    except Exception as e:
        logger.error("[BANANA] Falha ao abrir '%s': %s", arquivo_nome, e)

    vistos: set = set()
    unicos = []
    for r in registros:
        chave = (r["produto"], r["quant"], r["valor_total"])
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(r)
    return unicos


# ---------------------------------------------------------------------------
# 1. load_precos
# ---------------------------------------------------------------------------

def _load_precos_from_csv(pasta_precos: str = "") -> dict:
    """Carrega CSVs de preços de uma pasta."""
    resultado: dict = {}
    if not pasta_precos or not os.path.isdir(pasta_precos):
        logger.error("Pasta de preços inexistente: '%s'", pasta_precos)
        return resultado

    arquivos = glob.glob(os.path.join(pasta_precos, "preços_*.csv"))
    if not arquivos:
        arquivos = glob.glob(os.path.join(pasta_precos, "precos_*.csv"))
    if not arquivos:
        return resultado

    for caminho in arquivos:
        try:
            dt = parse_data_arquivo(caminho)
            if dt is None:
                continue
            chave = dt.strftime("%d-%m-%Y")
            df = pd.read_csv(caminho, encoding="utf-8", on_bad_lines="skip", engine="python")

            if df.shape[1] > 0:
                primeira_col = df.columns[0]
                df = df[~df[primeira_col].astype(str).str.contains("Busca gerada em", na=False)]

            df.columns = df.columns.str.strip()
            df.dropna(how="all", inplace=True)

            status_cols = [c for c in df.columns if c.lower().startswith("status")]
            if status_cols:
                mask_nenhuma_loja = pd.concat(
                    [df[c].astype(str).str.strip().str.lower().isin({"não encontrado", "sem match"})
                     for c in status_cols],
                    axis=1,
                ).all(axis=1)
                df = df[~mask_nenhuma_loja]

            for col in [c for c in df.columns if "preço" in c.lower() or "preco" in c.lower()]:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(r"[^\d,\.]", "", regex=True)
                    .str.replace(",", ".", regex=False)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

            if "Produto Buscado" in df.columns:
                df["Produto Buscado"] = df["Produto Buscado"].astype(str).str.strip().str.upper()

            df["Data"] = dt
            resultado[chave] = df
            logger.info("Preços carregados: %s → %d linhas.", chave, len(df))
        except Exception as exc:
            logger.error("Erro ao carregar preços '%s': %s", caminho, exc)

    return dict(sorted(
        resultado.items(),
        key=lambda item: datetime.strptime(item[0], "%d-%m-%Y"),
        reverse=True,
    ))


def load_precos(pasta_precos: str = "") -> dict:
    """Carrega precos do banco e usa CSVs apenas como fallback."""

    def normalizar_coluna(valor: str) -> str:
        import unicodedata

        texto = unicodedata.normalize("NFKD", str(valor))
        return texto.encode("ascii", "ignore").decode("ascii").lower().strip()

    def preparar_dataframe_precos(df: pd.DataFrame, dt: Optional[datetime]) -> pd.DataFrame:
        if df.shape[1] > 0:
            primeira_col = df.columns[0]
            df = df[~df[primeira_col].astype(str).str.contains("Busca gerada em", na=False)]

        df = df.copy()
        df.columns = df.columns.map(lambda coluna: str(coluna).strip())
        df.dropna(how="all", inplace=True)

        status_cols = [
            coluna for coluna in df.columns if normalizar_coluna(coluna).startswith("status")
        ]
        if status_cols:
            mask_nenhuma_loja = pd.concat(
                [
                    df[coluna]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .isin({"nao encontrado", "não encontrado", "sem match"})
                    for coluna in status_cols
                ],
                axis=1,
            ).all(axis=1)
            df = df[~mask_nenhuma_loja]

        for coluna in [
            coluna for coluna in df.columns if "preco" in normalizar_coluna(coluna)
        ]:
            df[coluna] = (
                df[coluna]
                .astype(str)
                .str.replace(r"[^\d,\.]", "", regex=True)
                .str.replace(",", ".", regex=False)
            )
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

        produto_col = next(
            (
                coluna
                for coluna in df.columns
                if "produto buscado" in normalizar_coluna(coluna)
            ),
            None,
        )
        if produto_col is None:
            produto_col = next(
                (coluna for coluna in df.columns if "produto" in normalizar_coluna(coluna)),
                None,
            )
        if produto_col is not None:
            df[produto_col] = df[produto_col].astype(str).str.strip().str.upper()

        if dt is not None:
            df["Data"] = dt

        return df

    def parse_data_preco(valor) -> Optional[datetime]:
        if valor is None or (isinstance(valor, float) and pd.isna(valor)):
            return None
        if isinstance(valor, datetime):
            return valor
        try:
            parsed = pd.to_datetime(valor, errors="coerce", dayfirst=True)
            if pd.notna(parsed):
                return parsed.to_pydatetime()
        except Exception:
            pass
        return parse_data_arquivo(str(valor))

    def encontrar_coluna_data(colunas: list[str]) -> Optional[str]:
        data_prioridades = {
            "data",
            "date",
            "data pesquisa",
            "data da pesquisa",
            "data coleta",
            "data consulta",
            "dt",
            "dt pesquisa",
            "data_pesquisa",
            "data_coleta",
            "created at",
            "created_at",
            "updated at",
            "updated_at",
            "imported at",
            "imported_at",
            "timestamp",
        }
        arquivo_prioridades = {
            "arquivo",
            "nome arquivo",
            "arquivo origem",
            "source file",
            "file name",
            "filename",
        }
        normalizadas = {coluna: normalizar_coluna(coluna) for coluna in colunas}

        for coluna, normalizada in normalizadas.items():
            if normalizada in data_prioridades:
                return coluna
        for coluna, normalizada in normalizadas.items():
            if normalizada.startswith("data "):
                return coluna
        for coluna, normalizada in normalizadas.items():
            if normalizada in arquivo_prioridades:
                return coluna
        return None

    try:
        registros_db = fetch_precos_rows()
    except Exception as exc:
        logger.warning("Falha ao consultar tabela de precos no banco: %s", exc)
        registros_db = []

    if registros_db:
        try:
            df_db = pd.DataFrame(registros_db)
            if not df_db.empty:
                coluna_data = encontrar_coluna_data(list(df_db.columns))
                if coluna_data:
                    resultado: dict[str, pd.DataFrame] = {}
                    grupos: dict[str, list[dict]] = {}
                    for registro in registros_db:
                        dt = parse_data_preco(registro.get(coluna_data))
                        if dt is None:
                            continue
                        chave = dt.strftime("%d-%m-%Y")
                        grupos.setdefault(chave, []).append(registro)

                    for chave, registros in grupos.items():
                        dt = datetime.strptime(chave, "%d-%m-%Y")
                        df_grupo = preparar_dataframe_precos(pd.DataFrame(registros), dt)
                        if not df_grupo.empty:
                            resultado[chave] = df_grupo

                    if resultado:
                        logger.info(
                            "Precos carregados do banco: %d data(s) consolidadas.",
                            len(resultado),
                        )
                        return dict(
                            sorted(
                                resultado.items(),
                                key=lambda item: datetime.strptime(item[0], "%d-%m-%Y"),
                                reverse=True,
                            )
                        )
                else:
                    logger.warning(
                        "Tabela de precos encontrada no banco, mas sem coluna de data reconhecida. Usando fallback em CSV."
                    )
                    df_unico = preparar_dataframe_precos(pd.DataFrame(registros_db), datetime.now())
                    if not df_unico.empty:
                        chave = datetime.now().strftime("%d-%m-%Y")
                        logger.info(
                            "Precos carregados do banco sem coluna de data; usando snapshot unico em %s.",
                            chave,
                        )
                        return {chave: df_unico}
        except Exception as exc:
            logger.warning("Falha ao montar precos a partir do banco: %s", exc)

    return _load_precos_from_csv(pasta_precos)


def listar_precos_consolidados(pasta_precos: str = "") -> list[dict]:
    """Retorna lista simplificada de precos com base no CSV mais recente."""
    dados = load_precos(pasta_precos)
    if not dados:
        return []

    _, df = next(iter(dados.items()))
    if df is None or df.empty:
        return []

    colunas = list(df.columns)
    if not colunas:
        return []

    def normalizar_coluna(valor: str) -> str:
        import unicodedata

        texto = unicodedata.normalize("NFKD", str(valor))
        return texto.encode("ascii", "ignore").decode("ascii").lower()

    produto_col = next((c for c in colunas if "produto" in normalizar_coluna(c)), None)
    if produto_col is None:
        produto_col = colunas[0]

    colunas_preco = [c for c in colunas if "preco" in normalizar_coluna(c)]
    mercado_col = next(
        (
            c
            for c in colunas
            if normalizar_coluna(c) in {"estabelecimento", "mercado", "loja", "concorrente"}
        ),
        None,
    )

    if mercado_col and any(normalizar_coluna(c) == "preco" for c in colunas_preco):
        preco_col = next(c for c in colunas_preco if normalizar_coluna(c) == "preco")
        resumo = df[[produto_col, mercado_col, preco_col]].copy()
        resumo[produto_col] = resumo[produto_col].astype(str).str.strip()
        resumo[mercado_col] = resumo[mercado_col].astype(str).str.strip()
        resumo[preco_col] = pd.to_numeric(resumo[preco_col], errors="coerce")
        resumo = resumo[(resumo[produto_col] != "") & (resumo[mercado_col] != "")]

        def eh_semar(valor: object) -> bool:
            return "semar" in normalizar_coluna(str(valor or ""))

        resultado = []
        for produto, grupo in resumo.groupby(produto_col, sort=True):
            grupo_com_preco = grupo[grupo[preco_col].notna()].copy()
            if grupo_com_preco.empty:
                continue

            semar_vals = [
                float(valor)
                for _, valor in grupo_com_preco.loc[
                    grupo_com_preco[mercado_col].map(eh_semar),
                    preco_col,
                ].items()
            ]
            if not semar_vals:
                continue

            concorrentes = grupo_com_preco.loc[
                ~grupo_com_preco[mercado_col].map(eh_semar),
                [mercado_col, preco_col],
            ]
            precos_concorrentes = [float(valor) for valor in concorrentes[preco_col].tolist()]
            mercados_cotados = {
                str(mercado).strip()
                for mercado in concorrentes[mercado_col].tolist()
                if str(mercado).strip()
            }

            resultado.append(
                {
                    "Produto": produto,
                    "Preco": round(sum(semar_vals) / len(semar_vals), 2),
                    "PrecoSemar": round(sum(semar_vals) / len(semar_vals), 2),
                    "PrecoConcorrentesMedia": (
                        round(sum(precos_concorrentes) / len(precos_concorrentes), 2)
                        if precos_concorrentes
                        else None
                    ),
                    "ConcorrentesCotados": len(mercados_cotados),
                }
            )

        return resultado

    preco_col = next((c for c in colunas_preco if "semar" in normalizar_coluna(c)), None)
    if preco_col is None and colunas_preco:
        preco_col = colunas_preco[0]
    if preco_col is None:
        colunas_numericas = [c for c in colunas if pd.api.types.is_numeric_dtype(df[c])]
        if colunas_numericas:
            preco_col = colunas_numericas[0]

    if preco_col is None:
        return []

    concorrente_cols = [
        c for c in colunas_preco if c != preco_col and "semar" not in normalizar_coluna(c)
    ]

    resumo = df[[produto_col, preco_col, *concorrente_cols]].copy()
    resumo[produto_col] = resumo[produto_col].astype(str).str.strip()
    resumo[preco_col] = pd.to_numeric(resumo[preco_col], errors="coerce")
    for col in concorrente_cols:
        resumo[col] = pd.to_numeric(resumo[col], errors="coerce")

    resumo = resumo[(resumo[produto_col] != "") & (resumo[preco_col].notna())]
    resumo = resumo.sort_values(produto_col, ascending=True).reset_index(drop=True)

    resultado = []
    for _, row in resumo.iterrows():
        precos_concorrentes = [
            float(row[col]) for col in concorrente_cols if pd.notna(row[col])
        ]
        media_concorrentes = (
            sum(precos_concorrentes) / len(precos_concorrentes)
            if precos_concorrentes
            else None
        )

        resultado.append(
            {
                "Produto": row[produto_col],
                "Preco": float(row[preco_col]),
                "PrecoSemar": float(row[preco_col]),
                "PrecoConcorrentesMedia": media_concorrentes,
                "ConcorrentesCotados": len(precos_concorrentes),
            }
        )

    return resultado


# ---------------------------------------------------------------------------
# 2. load_metas_vendas (mantido para compatibilidade)
# ---------------------------------------------------------------------------

def _excel_serial_to_datetime(serial) -> Optional[datetime]:
    try:
        return datetime(1899, 12, 30) + timedelta(days=int(serial))
    except Exception:
        return pd.NaT


def load_metas_vendas(arquivo_excel: str = "") -> tuple:
    """Carrega sheets Progresso/Pedidos/Metas do Excel legado."""
    if not arquivo_excel or not os.path.isfile(arquivo_excel):
        raise FileNotFoundError(f"Arquivo Excel não encontrado: '{arquivo_excel}'")

    xl = pd.ExcelFile(arquivo_excel, engine="openpyxl")
    faltando = {"Progresso", "Pedidos", "Metas"} - set(xl.sheet_names)
    if faltando:
        raise ValueError(f"Sheets ausentes: {faltando}")

    df_prog = xl.parse("Progresso")
    df_prog.columns = df_prog.columns.str.strip()
    df_prog.dropna(how="all", inplace=True)
    if "Produtos" in df_prog.columns:
        df_prog["Produtos"] = df_prog["Produtos"].astype(str).str.strip().str.upper()
    if "Progresso" in df_prog.columns:
        df_prog["Progresso"] = pd.to_numeric(
            df_prog["Progresso"].astype(str).str.replace("%", "").str.replace(",", ".").str.strip(),
            errors="coerce",
        )
    for col in ["meta", "pedido"]:
        if col in df_prog.columns:
            df_prog[col] = pd.to_numeric(
                df_prog[col].astype(str).str.replace(",", "."), errors="coerce"
            )

    df_ped = xl.parse("Pedidos", dtype=object)
    df_ped.columns = df_ped.columns.str.strip()
    df_ped.dropna(how="all", inplace=True)
    if "Data" in df_ped.columns:
        def _conv(val):
            if pd.isna(val):
                return pd.NaT
            if isinstance(val, datetime):
                return val
            try:
                return _excel_serial_to_datetime(float(str(val).strip()))
            except Exception:
                try:
                    return pd.to_datetime(val, dayfirst=True)
                except Exception:
                    return pd.NaT
        df_ped["Data"] = df_ped["Data"].apply(_conv)
    for col in ["QUANT", "VALOR TOTAL", "VALOR UNIT"]:
        if col in df_ped.columns:
            df_ped[col] = pd.to_numeric(
                df_ped[col].astype(str).str.replace(",", ".").str.replace(r"[^\d.\-]", "", regex=True),
                errors="coerce",
            )
    if "Personalizar" in df_ped.columns:
        df_ped["Personalizar"] = df_ped["Personalizar"].astype(str).str.strip().str.upper()
    if "Data" in df_ped.columns:
        df_ped = df_ped[df_ped["Data"].notna()]

    df_metas = xl.parse("Metas")
    df_metas.columns = df_metas.columns.str.strip()
    df_metas.dropna(how="all", inplace=True)
    if "produtos" in df_metas.columns:
        df_metas["produtos"] = df_metas["produtos"].astype(str).str.strip().str.upper()
    if "META" in df_metas.columns:
        df_metas["META"] = pd.to_numeric(
            df_metas["META"].astype(str).str.replace(",", "."), errors="coerce"
        ).fillna(0).astype(int)

    return df_prog, df_ped, df_metas


# ---------------------------------------------------------------------------
# 3. calcular_estoque
# ---------------------------------------------------------------------------

def calcular_estoque(
    pasta_entradas: str = "",
    pasta_saidas:   str = "",
    caminho_cache:  str = "",
    usar_threads:   bool = True,
) -> tuple:
    """Calcula saldo de estoque de bananas com cache incremental e multiprocessing."""
    cache     = _carregar_cache(caminho_cache, "cache_estoque")
    historico = []

    tarefas = []
    for pasta, tipo in [(pasta_entradas, "entrada"), (pasta_saidas, "saida")]:
        if not pasta or not os.path.isdir(pasta):
            continue
        for pdf in glob.glob(os.path.join(pasta, "*.pdf")):
            tarefas.append((pdf, tipo))

    tarefas_novas = []
    for caminho_pdf, tipo in tarefas:
        nome_arq  = os.path.basename(caminho_pdf)
        cache_key = f"{tipo}::{nome_arq}"
        if cache_key in cache:
            for reg in cache[cache_key].get("registros", []):
                historico.append({
                    "data":        datetime.fromisoformat(reg["data"]) if reg.get("data") else None,
                    "tipo":        tipo,
                    "produto":     reg["produto"],
                    "quant":       reg["quant"],
                    "unidade":     reg.get("unidade", "KG"),
                    "valor_unit":  reg.get("valor_unit", 0.0),
                    "valor_total": reg.get("valor_total", 0.0),
                    "arquivo":     nome_arq,
                })
        else:
            tarefas_novas.append((caminho_pdf, tipo))

    if tarefas_novas:
        N_WORKERS   = min(len(tarefas_novas), 5)
        SALVAR_CADA = 20
        logger.info("Estoque: %d PDF(s) novos com %d workers...", len(tarefas_novas), N_WORKERS)

        pendente: dict = {}
        concluidos = 0

        tarefas_sequencial: list = []  # fallback quando o pool crasha

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futuros = {executor.submit(_processar_pdf_worker, t): t for t in tarefas_novas}
            for futuro in as_completed(futuros):
                try:
                    itens = futuro.result()
                except BrokenProcessPool:
                    tarefa_falhou = futuros[futuro]
                    logger.warning(
                        "Pool encerrado ao processar '%s'. Restantes serão processados sequencialmente.",
                        tarefa_falhou[0],
                    )
                    tarefas_sequencial = [t for f, t in futuros.items() if not f.done()]
                    tarefas_sequencial.insert(0, tarefa_falhou)
                    break
                except Exception as exc:
                    logger.error("Erro worker estoque: %s", exc)
                    concluidos += 1
                    continue

                for item in itens:
                    nome_arq  = item["arquivo"]
                    tipo      = item["tipo"]
                    cache_key = f"{tipo}::{nome_arq}"
                    if cache_key not in pendente:
                        pendente[cache_key] = {"registros": []}
                    pendente[cache_key]["registros"].append({
                        "data":        item["data"].isoformat() if item.get("data") else None,
                        "produto":     item["produto"],
                        "quant":       item["quant"],
                        "unidade":     item.get("unidade", "KG"),
                        "valor_unit":  item.get("valor_unit", 0.0),
                        "valor_total": item.get("valor_total", 0.0),
                    })
                    historico.append(item)

                concluidos += 1
                if concluidos % SALVAR_CADA == 0:
                    cache.update(pendente)
                    _salvar_cache(cache, caminho_cache, "cache_estoque")
                    pendente = {}
                    logger.info("Estoque: %d/%d PDFs.", concluidos, len(tarefas_novas))

        # Fallback sequencial para tarefas não processadas após crash do pool
        for tarefa in tarefas_sequencial:
            try:
                itens = _processar_pdf_worker(tarefa)
            except Exception as exc:
                logger.error("Erro sequencial estoque '%s': %s", tarefa[0], exc)
                concluidos += 1
                continue

            for item in itens:
                nome_arq  = item["arquivo"]
                tipo      = item["tipo"]
                cache_key = f"{tipo}::{nome_arq}"
                if cache_key not in pendente:
                    pendente[cache_key] = {"registros": []}
                pendente[cache_key]["registros"].append({
                    "data":        item["data"].isoformat() if item.get("data") else None,
                    "produto":     item["produto"],
                    "quant":       item["quant"],
                    "unidade":     item.get("unidade", "KG"),
                    "valor_unit":  item.get("valor_unit", 0.0),
                    "valor_total": item.get("valor_total", 0.0),
                })
                historico.append(item)
            concluidos += 1

        if pendente:
            cache.update(pendente)
            _salvar_cache(cache, caminho_cache, "cache_estoque")

    manuais = fetch_movimentacoes()
    for manual in manuais:
        data_manual = manual.get("data")
        if isinstance(data_manual, str):
            try:
                data_manual = datetime.fromisoformat(data_manual)
            except ValueError:
                data_manual = None
        elif not isinstance(data_manual, datetime):
            data_manual = None

        quant_manual = _parse_numero(manual.get("quant"))
        historico.append(
            {
                "data": data_manual,
                "tipo": str(manual.get("tipo") or "entrada").strip().lower(),
                "produto": str(manual.get("produto") or "").strip().upper(),
                "quant": quant_manual if quant_manual is not None else 0.0,
                "unidade": str(manual.get("unidade") or "KG").strip().upper(),
                "valor_unit": _parse_numero(manual.get("valor_unit")) or 0.0,
                "valor_total": _parse_numero(manual.get("valor_total")) or 0.0,
                "arquivo": "manual",
                "loja": str(manual.get("loja") or "").strip(),
            }
        )

    historico.sort(key=lambda x: (x["data"] is None, x["data"] or datetime.min, x["tipo"]))
    saldo = sum(i["quant"] if i["tipo"] == "entrada" else -i["quant"] for i in historico)
    logger.info("Saldo: %.2f kg | %d registros.", saldo, len(historico))
    return saldo, historico


# ---------------------------------------------------------------------------
# 4. load_pedidos_pdfs
# ---------------------------------------------------------------------------

def load_pedidos_pdfs(pasta_pdfs: str = "", caminho_cache: str = "") -> pd.DataFrame:
    """Carrega NF-e PDFs usando cache_pedidos relacional."""
    if not pasta_pdfs:
        logger.warning("Pasta de pedidos NF-e não informada.")
        return pd.DataFrame()
    if not os.path.isdir(pasta_pdfs):
        try:
            os.makedirs(pasta_pdfs, exist_ok=True)
        except Exception:
            pass
        logger.warning("Pasta de pedidos NF-e não encontrada: '%s'", pasta_pdfs)
        return pd.DataFrame()

    cache     = fetch_cache("cache_pedidos")
    registros = []

    pdfs = glob.glob(os.path.join(pasta_pdfs, "*.pdf"))
    if not pdfs:
        return pd.DataFrame()

    pdfs_novos = []
    for caminho_pdf in pdfs:
        nome_arq  = os.path.basename(caminho_pdf)
        cache_key = f"pedido::{nome_arq}"
        if cache_key in cache:
            for reg in cache[cache_key]:
                registros.append({
                    "Data":        datetime.fromisoformat(reg["data"]) if reg.get("data") else None,
                    "Loja":        reg["loja"],
                    "Localizacao": reg.get("localizacao", ""),
                    "Produto":     reg["produto"],
                    "UNID":        reg["unidade"],
                    "QUANT":       reg["quant"],
                    "VALOR TOTAL": reg["valor_total"],
                    "VALOR UNIT":  reg["valor_unit"],
                })
        else:
            pdfs_novos.append(caminho_pdf)

    if pdfs_novos:
        # E5-2620 v3: 6 núcleos / 12 threads — 10 workers é o ponto ótimo
        N_WORKERS   = min(len(pdfs_novos), 10)
        SALVAR_CADA = 25
        logger.info("Processando %d PDF(s) novo(s) com %d workers...", len(pdfs_novos), N_WORKERS)

        registros_para_persistir: list[dict] = []
        concluidos = 0

        pdfs_sequencial: list = []  # fallback quando o pool crasha

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futuros = {executor.submit(_worker_pedido, p): p for p in pdfs_novos}
            for futuro in as_completed(futuros):
                try:
                    nome_arq, dt, loja, localizacao, produtos = futuro.result()
                except BrokenProcessPool:
                    caminho_falhou = futuros[futuro]
                    logger.warning(
                        "Pool encerrado ao processar '%s'. Restantes serão processados sequencialmente.",
                        caminho_falhou,
                    )
                    pdfs_sequencial = [p for f, p in futuros.items() if not f.done()]
                    pdfs_sequencial.insert(0, caminho_falhou)
                    break
                except Exception as exc:
                    logger.error("Erro ao processar '%s': %s", futuros[futuro], exc)
                    concluidos += 1
                    continue

                cache_key = f"pedido::{nome_arq}"
                for p in produtos:
                    registros.append({
                        "Data":        dt,
                        "Loja":        loja,
                        "Localizacao": localizacao,
                        "Produto":     p["produto"],
                        "UNID":        p["unidade"],
                        "QUANT":       p["quant"],
                        "VALOR TOTAL": p["valor_total"],
                        "VALOR UNIT":  p["valor_unit"],
                    })
                    registros_para_persistir.append({
                        "data":        dt.isoformat() if dt else None,
                        "loja":        loja,
                        "produto":     p["produto"],
                        "unidade":     p["unidade"],
                        "quant":       p["quant"],
                        "valor_total": p["valor_total"],
                        "valor_unit":  p["valor_unit"],
                        "arquivo_pdf": cache_key,
                    })

                concluidos += 1
                if concluidos % SALVAR_CADA == 0:
                    replace_cache_pedidos(registros_para_persistir)
                    registros_para_persistir = []
                    logger.info(
                        "Progresso: %d/%d PDFs | cache_pedidos relacional atualizado.",
                        concluidos,
                        len(pdfs_novos),
                    )

        # Fallback sequencial para PDFs não processados após crash do pool
        for caminho_pdf in pdfs_sequencial:
            try:
                nome_arq, dt, loja, localizacao, produtos = _worker_pedido(caminho_pdf)
            except Exception as exc:
                logger.error("Erro sequencial '%s': %s", caminho_pdf, exc)
                concluidos += 1
                continue

            cache_key = f"pedido::{nome_arq}"
            for p in produtos:
                registros.append({
                    "Data":        dt,
                    "Loja":        loja,
                    "Localizacao": localizacao,
                    "Produto":     p["produto"],
                    "UNID":        p["unidade"],
                    "QUANT":       p["quant"],
                    "VALOR TOTAL": p["valor_total"],
                    "VALOR UNIT":  p["valor_unit"],
                })
                registros_para_persistir.append({
                    "data":        dt.isoformat() if dt else None,
                    "loja":        loja,
                    "produto":     p["produto"],
                    "unidade":     p["unidade"],
                    "quant":       p["quant"],
                    "valor_total": p["valor_total"],
                    "valor_unit":  p["valor_unit"],
                    "arquivo_pdf": cache_key,
                })

            concluidos += 1

        if registros_para_persistir:
            replace_cache_pedidos(registros_para_persistir)

        logger.info("Concluído: %d/%d PDFs.", concluidos, len(pdfs_novos))

    if not registros:
        return pd.DataFrame()

    df = pd.DataFrame(registros)
    df["Data"]        = pd.to_datetime(df["Data"],       errors="coerce")
    df["QUANT"]       = pd.to_numeric(df["QUANT"],       errors="coerce")
    df["VALOR TOTAL"] = pd.to_numeric(df["VALOR TOTAL"], errors="coerce")
    df["VALOR UNIT"]  = pd.to_numeric(df["VALOR UNIT"],  errors="coerce")
    df["Produto"]     = df["Produto"].astype(str).str.strip().str.upper()
    return df.sort_values("Data", ascending=False).reset_index(drop=True)


def _serializar_data_job(valor) -> str | None:
    if valor is None or (hasattr(pd, "isna") and pd.isna(valor)):
        return None
    if isinstance(valor, datetime):
        return valor.isoformat()
    try:
        convertido = pd.to_datetime(valor, errors="coerce")
    except Exception:
        convertido = None
    if convertido is None or pd.isna(convertido):
        return str(valor)
    if hasattr(convertido, "to_pydatetime"):
        convertido = convertido.to_pydatetime()
    return convertido.isoformat()


def _arquivo_eh_pedido_semar(caminho_pdf: str) -> bool:
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            texto = (pdf.pages[0].extract_text() or "").lower() if pdf.pages else ""
            return "pedido de compra" in texto
    except Exception:
        return False


def _prefixar_arquivo_upload(nome_arquivo: str, *, semar: bool) -> str:
    prefixo = "semar::" if semar else "pedido::"
    return f"{prefixo}{os.path.basename(nome_arquivo)}"


def _carregar_registros_upload_pdf(caminho_pdf: str) -> list[dict]:
    eh_semar = _arquivo_eh_pedido_semar(caminho_pdf)
    arquivo_cache = _prefixar_arquivo_upload(caminho_pdf, semar=eh_semar)

    if eh_semar:
        df = extrair_pedido_semar(caminho_pdf)
        if df.empty:
            return []

        registros: list[dict] = []
        for _, row in df.iterrows():
            registros.append(
                {
                    "Data": _serializar_data_job(row.get("Data")),
                    "Loja": str(row.get("Loja") or "").strip(),
                    "Produto": str(row.get("Produto") or "").strip().upper(),
                    "UNID": str(row.get("UNID") or "KG").strip().upper(),
                    "QUANT": float(pd.to_numeric(row.get("QUANT"), errors="coerce") or 0),
                    "VALOR TOTAL": float(
                        pd.to_numeric(row.get("VALOR TOTAL"), errors="coerce") or 0
                    ),
                    "VALOR UNIT": float(
                        pd.to_numeric(row.get("VALOR UNIT"), errors="coerce") or 0
                    ),
                    "ARQUIVO": arquivo_cache,
                }
            )
        return registros

    nome_arq, dt, loja, _localizacao, produtos = _worker_pedido(caminho_pdf)
    registros: list[dict] = []
    for produto in produtos:
        registros.append(
            {
                "Data": _serializar_data_job(dt),
                "Loja": str(loja or "").strip(),
                "Produto": str(produto.get("produto") or "").strip().upper(),
                "UNID": str(produto.get("unidade") or "KG").strip().upper(),
                "QUANT": float(produto.get("quant") or 0),
                "VALOR TOTAL": float(produto.get("valor_total") or 0),
                "VALOR UNIT": float(produto.get("valor_unit") or 0),
                "ARQUIVO": arquivo_cache,
            }
        )
    return registros


def processar_pedidos_upload(
    pdf_paths: list[str], progress_callback: Optional[Callable[[dict], None]] = None
) -> dict:
    registros: list[dict] = []
    total_files = len(pdf_paths)

    for indice, caminho_pdf in enumerate(pdf_paths, start=1):
        registros_arquivo = _carregar_registros_upload_pdf(caminho_pdf)
        registros.extend(registros_arquivo)

        if progress_callback is not None:
            progress_callback(
                {
                    "processed_files": indice,
                    "saved_records": len(registros),
                    "current_file": os.path.basename(caminho_pdf),
                    "records_added": len(registros_arquivo),
                    "total_files": total_files,
                }
            )

    replace_cache_pedidos(registros)
    return {
        "processed_files": total_files,
        "saved_records": len(registros),
        "total_files": total_files,
    }


def _append_import_job_log(job_id: str, message: str) -> None:
    job = get_import_job(job_id)
    if job is None:
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    recent_logs = list(job.get("recent_logs") or [])
    recent_logs.append(f"[{timestamp}] {message}")
    update_import_job(job_id, recent_logs=recent_logs[-IMPORT_JOB_LOG_LIMIT:])


def _heartbeat_import_job(job_id: str, current_file: str, stop_event: threading.Event) -> None:
    while not stop_event.wait(IMPORT_JOB_HEARTBEAT_SECONDS):
        update_import_job(
            job_id,
            status="processing",
            current_file=current_file,
            touch_heartbeat=True,
        )


def run_import_job(job_id: str, pdf_paths: list[str], working_dir: str) -> None:
    total_files = len(pdf_paths)
    registros: list[dict] = []
    processed_files = 0

    try:
        update_import_job(
            job_id,
            status="processing",
            total_files=total_files,
            processed_files=0,
            saved_records=0,
            current_file=None,
            error_message=None,
            touch_heartbeat=True,
        )
        _append_import_job_log(job_id, f"Importacao iniciada com {total_files} arquivo(s).")

        for caminho_pdf in pdf_paths:
            current_file = os.path.basename(caminho_pdf)
            update_import_job(
                job_id,
                status="processing",
                current_file=current_file,
                touch_heartbeat=True,
            )
            _append_import_job_log(job_id, f"Processando {current_file}.")

            stop_event = threading.Event()
            heartbeat_thread = threading.Thread(
                target=_heartbeat_import_job,
                args=(job_id, current_file, stop_event),
                daemon=True,
            )
            heartbeat_thread.start()
            try:
                registros_arquivo = _carregar_registros_upload_pdf(caminho_pdf)
            finally:
                stop_event.set()
                heartbeat_thread.join(timeout=1)

            registros.extend(registros_arquivo)
            processed_files += 1

            update_import_job(
                job_id,
                status="processing",
                processed_files=processed_files,
                saved_records=len(registros),
                current_file=current_file,
                touch_heartbeat=True,
            )
            _append_import_job_log(
                job_id,
                f"Concluido {current_file}: {len(registros_arquivo)} registro(s).",
            )

        replace_cache_pedidos(registros)
        update_import_job(
            job_id,
            status="completed",
            processed_files=processed_files,
            saved_records=len(registros),
            current_file=None,
            error_message=None,
            touch_heartbeat=True,
            finished=True,
        )
        _append_import_job_log(
            job_id,
            f"Importacao concluida com {processed_files} arquivo(s) e {len(registros)} registro(s).",
        )
    except Exception as exc:
        logger.exception("Falha ao executar job de importacao %s", job_id)
        update_import_job(
            job_id,
            status="failed",
            processed_files=processed_files,
            saved_records=len(registros),
            current_file=None,
            error_message=str(exc).strip() or "Falha ao processar importacao.",
            touch_heartbeat=True,
            finished=True,
        )
        _append_import_job_log(job_id, f"Falha na importacao: {exc}")
    finally:
        try:
            shutil.rmtree(working_dir, ignore_errors=True)
        except Exception:
            logger.warning("Nao foi possivel limpar diretorio temporario '%s'.", working_dir)


# ---------------------------------------------------------------------------
# 5. Metas locais
# ---------------------------------------------------------------------------

def load_metas_local(caminho_json: str) -> pd.DataFrame:
    lista = load_metas()
    if not lista:
        return pd.DataFrame(columns=["Produto", "Meta"])
    df = pd.DataFrame(lista)
    if df.empty or "Produto" not in df.columns or "Meta" not in df.columns:
        return pd.DataFrame(columns=["Produto", "Meta"])
    df["Produto"] = df["Produto"].astype(str).str.strip().str.upper()
    df["Meta"]    = pd.to_numeric(df["Meta"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def salvar_metas_local(lista_metas: list, caminho_json: str) -> None:
    try:
        replace_metas(lista_metas)
        logger.info("Metas salvas: %d item(s).", len(lista_metas))
    except Exception as exc:
        logger.error("Falha ao salvar metas '%s': %s", caminho_json, exc)


# ---------------------------------------------------------------------------
# 6. Movimentações manuais (registro avulso de estoque)
# ---------------------------------------------------------------------------

def salvar_movimentacao_manual(registros: list, caminho_json: str) -> None:
    """Persiste registros no banco (semelhante ao append anterior)."""
    agora = datetime.now().isoformat()
    for reg in registros:
        reg.setdefault("data", agora)
        reg.setdefault("arquivo", "manual")
        reg.setdefault("unidade", "KG")
        reg.setdefault("loja", "")
    try:
        insert_movimentacoes(registros)
        logger.info("Movimentações manuais: %d registro(s) salvos.", len(registros))
    except Exception as exc:
        logger.error("Falha ao salvar movimentações manuais '%s': %s", caminho_json, exc)


def load_movimentacoes_manuais(caminho_json: str) -> list:
    """Carrega lista de movimentações manuais persistidas no banco."""
    try:
        return fetch_movimentacoes()
    except Exception as exc:
        logger.error("Falha ao carregar movimentações manuais '%s': %s", caminho_json, exc)
        return []


def deletar_movimentacao_manual(entry_id: int, caminho_json: str) -> None:
    """Remove o registro identificado pelo ID."""
    try:
        delete_movimentacao(entry_id)
    except Exception as exc:
        logger.error("Falha ao deletar movimentação manual '%s': %s", caminho_json, exc)


# ---------------------------------------------------------------------------
# 7. Extrator de bananas de NF-e para upload manual
# ---------------------------------------------------------------------------

def extrair_bananas_pdf_upload(caminho_pdf: str, nome_original: str = "") -> list:
    """Extrai apenas linhas de BANANA de um DANFE (uso na página de registro).

    Returns:
        Lista de dicts com {produto, quant, unidade, valor_unit, valor_total}
    """
    return _extrair_bananas_pdf(caminho_pdf)


# ---------------------------------------------------------------------------
# 8. Registro de caixas por loja
# ---------------------------------------------------------------------------

_DEFAULT_CAIXAS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dados", "cache", "caixas_lojas.json")


def salvar_registro_caixas(registro: dict, caminho_json: str = _DEFAULT_CAIXAS_JSON) -> None:
    """Salva registro de caixas no banco."""
    try:
        insert_caixa(registro)
        logger.info("Caixas: registro salvo para loja %s.", registro.get("loja", "?"))
    except Exception as exc:
        logger.error("Falha ao salvar caixas '%s': %s", caminho_json, exc)


def load_registros_caixas(caminho_json: str = _DEFAULT_CAIXAS_JSON) -> pd.DataFrame:
    """Carrega todos os registros de caixas. Retorna DataFrame vazio se não existir."""
    _colunas = [
        "data",
        "loja",
        "n_loja",
        "caixas_benverde",
        "caixas_ccj",
        "ccj_banca",
        "ccj_mercadoria",
        "ccj_retirada",
        "caixas_bananas",
        "total",
        "entregue",
    ]
    try:
        registros = fetch_caixas()
        if not registros:
            return pd.DataFrame(columns=_colunas)
        df = pd.DataFrame(registros)
        for col in _colunas:
            if col not in df.columns:
                df[col] = None
        df = df[_colunas].copy()
        df["n_loja"] = pd.to_numeric(df["n_loja"], errors="coerce").fillna(0).astype(int)
        for col in [
            "caixas_benverde",
            "caixas_ccj",
            "ccj_banca",
            "ccj_mercadoria",
            "ccj_retirada",
            "caixas_bananas",
            "total",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        return df
    except Exception as exc:
        logger.error("Falha ao carregar caixas '%s': %s", caminho_json, exc)
        return pd.DataFrame(columns=_colunas)


# ---------------------------------------------------------------------------
# 9. Resumos para a IA
# ---------------------------------------------------------------------------

def resumo_precos_para_prompt(dados_precos: dict, max_dias: int = 3) -> str:
    if not dados_precos:
        return "Nenhum dado de preços disponível."
    linhas = []
    for data_str, df in list(dados_precos.items())[:max_dias]:
        linhas.append(f"\n=== Preços em {data_str} ===")
        linhas.append(df.to_string(index=False, max_rows=50))
    return "\n".join(linhas)


def resumo_estoque_para_prompt(saldo: float, historico: list, ultimos_n: int = 20) -> str:
    linhas = [f"Saldo atual de bananas: {saldo:.2f} kg\n", "Últimas movimentações:"]
    for item in historico[-ultimos_n:]:
        data_fmt = item["data"].strftime("%d/%m/%Y") if item["data"] else "?"
        linhas.append(
            f"  [{item['tipo'].upper()}] {data_fmt} | {item['produto']} | "
            f"{item['quant']:.2f} {item['unidade']} | {item['arquivo']}"
        )
    return "\n".join(linhas)


# ---------------------------------------------------------------------------
# 10. Pedidos Semar — formato compatível com load_pedidos_pdfs
# ---------------------------------------------------------------------------

def extrair_pedido_semar(caminho_pdf: str) -> pd.DataFrame:
    """Extrai Pedido de Compra Semar e retorna no mesmo formato de load_pedidos_pdfs().

    Colunas retornadas (idênticas ao DataFrame de NF-e para concatenação direta):
      Data (datetime) | Loja (str) | Produto (str) | UNID (str) |
      QUANT (float) | VALOR TOTAL (float) | VALOR UNIT (float)

    Estrutura real do PDF (página 1):
      Uma única tabela grande com todas os produtos. Por produto:
        - Linha de cabeçalho  : col 0 = "BANANA X kg - Embalagem..."
        - Linha de custo      : col 0 = "Custo Emb. Custo Unit. ...\\nV1 V2 ..." (col2=Custo Unit.)
        - Linha(s) de lojas   : cols 1-N = "LOJA 10 - TAUBATE" etc.
        - Linha(s) de qtds    : cols 1-N = quantidades numéricas
        - Linha "Total:"      : ignorada
      Lojas com mais de 12 aparecem em grupos adicionais (overflow).
    """
    _colunas = ["Data", "Loja", "Produto", "UNID", "QUANT", "VALOR TOTAL", "VALOR UNIT"]
    arquivo_nome = os.path.basename(caminho_pdf)
    registros: list = []

    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            # Data de emissão (página 1)
            texto_p1 = pdf.pages[0].extract_text() or ""
            data_pedido = None
            m_data = _RE_DATA_SEMAR.search(texto_p1)
            if m_data:
                try:
                    data_pedido = datetime.strptime(m_data.group(1), "%d/%m/%Y")
                except ValueError:
                    pass

            # Processar apenas página 1 — produtos/lojas/quantidades estão aqui.
            # Página 2 contém totais e endereços de entrega (não usados aqui).
            for tabela in (pdf.pages[0].extract_tables() or []):
                if not tabela:
                    continue
                # Só processar tabelas com pelo menos 3 colunas
                n_cols = max((len(r) for r in tabela if r), default=0)
                if n_cols < 3:
                    continue

                produto_atual = None
                custo_atual   = 0.0
                lojas_atuais: dict = {}   # col_index → nome normalizado da loja

                for row in tabela:
                    if not row:
                        continue
                    col0 = str(row[0] or "").strip()

                    # ── Cabeçalho de produto ──────────────────────────────────
                    # Ex: "BANANA NANICA kg - Embalagem com 1.0 KG * ..."
                    m_prod = _RE_PROD_SEMAR.match(col0)
                    if m_prod:
                        produto_atual = m_prod.group(1).strip().upper()
                        custo_atual   = 0.0
                        lojas_atuais  = {}
                        continue

                    # ── Linha de custo ────────────────────────────────────────
                    # Ex: "Custo Emb. Custo Unit. ...\n2,7000 2,7000 0,0000 ..."
                    # O 2º número é o Custo Unit.
                    if "custo unit" in col0.lower():
                        nums = re.findall(r'(\d+[.,]\d+)', col0)
                        if len(nums) >= 2:
                            custo_atual = _parse_br(nums[1])
                        elif len(nums) == 1:
                            custo_atual = _parse_br(nums[0])
                        continue

                    # ── Cabeçalho de lojas ────────────────────────────────────
                    # Qualquer célula (exceto col 0) contém "LOJA \d+"
                    celulas = [str(c or "").strip() for c in row]
                    if any(_RE_LOJA_SEMAR.search(c) for c in celulas[1:]):
                        lojas_atuais = {}
                        for idx, cell in enumerate(row):
                            cell_str = str(cell or "").strip()
                            if _RE_LOJA_SEMAR.search(cell_str):
                                # Normaliza quebras: "LOJA 10 -\nTAUBATE" → "LOJA 10 - TAUBATE"
                                lojas_atuais[idx] = " ".join(cell_str.split())
                        continue

                    # ── Linha de quantidades ──────────────────────────────────
                    if produto_atual and lojas_atuais:
                        for idx, cell in enumerate(row):
                            val = str(cell or "").strip()
                            if not val or not _RE_QUANT_SEMAR.match(val):
                                continue
                            quant = _parse_br(val)
                            if quant <= 0 or idx not in lojas_atuais:
                                continue
                            registros.append({
                                "Data":        data_pedido,
                                "Loja":        lojas_atuais[idx],
                                "Produto":     produto_atual,
                                "UNID":        "KG",
                                "QUANT":       quant,
                                "VALOR TOTAL": round(quant * custo_atual, 2),
                                "VALOR UNIT":  custo_atual,
                            })

    except Exception as exc:
        logger.error("[SEMAR] Falha ao processar '%s': %s", arquivo_nome, exc)

    if not registros:
        return pd.DataFrame(columns=_colunas)

    df = pd.DataFrame(registros)
    df = df.drop_duplicates(subset=["Produto", "Loja", "QUANT"])
    df["Data"]        = pd.to_datetime(df["Data"],       errors="coerce")
    df["QUANT"]       = pd.to_numeric(df["QUANT"],       errors="coerce")
    df["VALOR TOTAL"] = pd.to_numeric(df["VALOR TOTAL"], errors="coerce")
    df["VALOR UNIT"]  = pd.to_numeric(df["VALOR UNIT"],  errors="coerce")
    df["Produto"]     = df["Produto"].astype(str).str.strip().str.upper()
    logger.info("[SEMAR] '%s' → %d linha(s).", arquivo_nome, len(df))
    return df.reset_index(drop=True)


def load_pedidos_semar(pasta: str, caminho_cache: str = "") -> pd.DataFrame:
    """Varre pasta de PDFs Semar usando cache_pedidos relacional.

    Usa o mesmo padrão de cache de load_pedidos_pdfs():
      cache_key = f"semar::{nome_arquivo}"

    Retorna DataFrame com as mesmas colunas de load_pedidos_pdfs()
    (Data | Loja | Produto | UNID | QUANT | VALOR TOTAL | VALOR UNIT).
    Retorna DataFrame vazio se pasta inválida ou sem PDFs.
    """
    _colunas = ["Data", "Loja", "Produto", "UNID", "QUANT", "VALOR TOTAL", "VALOR UNIT"]

    if not pasta:
        logger.warning("[SEMAR] Pasta não informada.")
        return pd.DataFrame(columns=_colunas)
    if not os.path.isdir(pasta):
        try:
            os.makedirs(pasta, exist_ok=True)
        except Exception:
            pass
        logger.warning("[SEMAR] Pasta inválida ou inexistente: '%s'", pasta)
        return pd.DataFrame(columns=_colunas)

    pdfs = glob.glob(os.path.join(pasta, "*.pdf"))
    if not pdfs:
        return pd.DataFrame(columns=_colunas)

    cache     = fetch_cache("cache_pedidos")
    registros: list = []
    pdfs_novos: list = []

    for caminho_pdf in pdfs:
        nome_arq  = os.path.basename(caminho_pdf)
        cache_key = f"semar::{nome_arq}"
        if cache_key in cache:
            for reg in cache[cache_key]:
                registros.append({
                    "Data":        datetime.fromisoformat(reg["data"]) if reg.get("data") else None,
                    "Loja":        reg["loja"],
                    "Produto":     reg["produto"],
                    "UNID":        reg.get("unidade", "KG"),
                    "QUANT":       reg["quant"],
                    "VALOR TOTAL": reg["valor_total"],
                    "VALOR UNIT":  reg["valor_unit"],
                })
        else:
            pdfs_novos.append(caminho_pdf)

    if pdfs_novos:
        registros_para_persistir: list[dict] = []
        for caminho_pdf in pdfs_novos:
            nome_arq  = os.path.basename(caminho_pdf)
            cache_key = f"semar::{nome_arq}"
            df_pdf    = extrair_pedido_semar(caminho_pdf)
            for _, row in df_pdf.iterrows():
                registros.append({
                    "Data":        row["Data"],
                    "Loja":        row["Loja"],
                    "Produto":     row["Produto"],
                    "UNID":        row["UNID"],
                    "QUANT":       row["QUANT"],
                    "VALOR TOTAL": row["VALOR TOTAL"],
                    "VALOR UNIT":  row["VALOR UNIT"],
                })
                registros_para_persistir.append({
                    "data":        row["Data"].isoformat() if pd.notna(row["Data"]) else None,
                    "loja":        row["Loja"],
                    "produto":     row["Produto"],
                    "unidade":     row["UNID"],
                    "quant":       row["QUANT"],
                    "valor_total": row["VALOR TOTAL"],
                    "valor_unit":  row["VALOR UNIT"],
                    "arquivo_pdf": cache_key,
                })
        replace_cache_pedidos(registros_para_persistir)
        logger.info("[SEMAR] %d PDF(s) novos processados.", len(pdfs_novos))

    if not registros:
        return pd.DataFrame(columns=_colunas)

    df = pd.DataFrame(registros)
    df["Data"]        = pd.to_datetime(df["Data"],       errors="coerce")
    df["QUANT"]       = pd.to_numeric(df["QUANT"],       errors="coerce")
    df["VALOR TOTAL"] = pd.to_numeric(df["VALOR TOTAL"], errors="coerce")
    df["VALOR UNIT"]  = pd.to_numeric(df["VALOR UNIT"],  errors="coerce")
    df["Produto"]     = df["Produto"].astype(str).str.strip().str.upper()
    return df.sort_values("Data", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bloco de testes
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    PASTA_PRECOS   = r"C:\Users\pesso\OneDrive\Documentos\benverde\MeuAppGerencia\dados\precos"
    PASTA_ENTRADAS = r"C:\Users\pesso\OneDrive\Documentos\benverde\MeuAppGerencia\dados\entradas_bananas"
    PASTA_SAIDAS   = r"C:\Users\pesso\OneDrive\Documentos\benverde\MeuAppGerencia\dados\saidas_bananas"

    print("=" * 60)
    print("TESTE 1: _parse_nome_arquivo_nfe")
    print("=" * 60)
    exemplos = [
        "230201.pdf", "01.01 lj21.pdf", "0102_loja04.pdf",
        "2301 libra.pdf", "10 01 1.pdf", "10 01 lj11.pdf",
        "10,01 17.pdf", "27j,an 18.pdf", "28, 01 libra.pdf",
    ]
    for ex in exemplos:
        dt, loja = _parse_nome_arquivo_nfe(ex)
        print(f"  {ex:30s} → {dt.strftime('%d/%m/%Y') if dt else 'None':12s} | {loja}")

    print("\n" + "=" * 60)
    print("TESTE 2: load_precos")
    print("=" * 60)
    precos = load_precos(PASTA_PRECOS)
    for data, df in list(precos.items())[:2]:
        print(f"\n{data} | {df.shape}")
        print(df.head(3).to_string())
