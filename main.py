from dotenv import load_dotenv

load_dotenv()

import logging
from multiprocessing import Process
import os
import re
import shutil
import unicodedata
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

from fastapi import (
    Body,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api_auth import get_current_user, router as auth_router
from data_processor import (
    calcular_estoque,
    deletar_movimentacao_manual,
    extrair_bananas_pdf_upload,
    listar_precos_consolidados,
    load_precos,
    load_movimentacoes_manuais,
    load_registros_caixas,
    run_import_job,
    salvar_movimentacao_manual,
    salvar_registro_caixas,
)
from db import (
    create_import_job,
    get_import_job,
    replace_cache_pedidos,
    update_import_job,
)
from services.mita_ai import chat_with_mita

logger = logging.getLogger(__name__)
IMPORT_JOBS_ROOT = Path(__file__).resolve().parent / "temp_import_jobs"

app = FastAPI(title="Benverde API")


class MitaHistoryMessage(BaseModel):
    role: str
    content: str = ""


class MitaChatPayload(BaseModel):
    message: str = Field(default="")
    history: list[MitaHistoryMessage] = Field(default_factory=list)


class ProcessedPedidoRecord(BaseModel):
    Data: str | None = None
    Loja: str = Field(default="")
    Produto: str = Field(default="")
    UNID: str = Field(default="KG")
    QUANT: float = Field(default=0.0)
    VALOR_TOTAL: float = Field(default=0.0, alias="VALOR TOTAL")
    VALOR_UNIT: float = Field(default=0.0, alias="VALOR UNIT")
    ARQUIVO: str = Field(default="")

    def to_storage_dict(self) -> dict:
        return {
            "Data": self.Data,
            "Loja": self.Loja.strip(),
            "Produto": self.Produto.strip().upper(),
            "UNID": (self.UNID or "KG").strip().upper(),
            "QUANT": float(self.QUANT or 0.0),
            "VALOR TOTAL": float(self.VALOR_TOTAL or 0.0),
            "VALOR UNIT": float(self.VALOR_UNIT or 0.0),
            "ARQUIVO": Path(self.ARQUIVO or "").name.strip(),
        }


class ProcessedPedidosPayload(BaseModel):
    registros: list[ProcessedPedidoRecord] = Field(default_factory=list)
    arquivos: list[str] = Field(default_factory=list)
    total_arquivos: int | None = None
    origem: str = Field(default="app_local")


def _serialize_import_job(job: dict) -> dict:
    total_files = max(0, int(job.get("total_files") or 0))
    processed_files = max(0, int(job.get("processed_files") or 0))
    saved_records = max(0, int(job.get("saved_records") or 0))
    remaining_files = max(0, total_files - processed_files)

    started_at = job.get("started_at")
    finished_at = job.get("finished_at")
    now = datetime.now(started_at.tzinfo) if started_at is not None else datetime.now()
    reference_time = finished_at or now
    elapsed_seconds = None
    if started_at is not None:
        elapsed_seconds = max(0, int((reference_time - started_at).total_seconds()))

    eta_seconds = None
    if processed_files > 0 and elapsed_seconds is not None:
        media_por_arquivo = elapsed_seconds / processed_files
        eta_seconds = max(0, int(media_por_arquivo * remaining_files))
        if job.get("status") == "completed":
            eta_seconds = 0

    progress_percent = 0.0
    if total_files > 0:
        progress_percent = min(100.0, (processed_files / total_files) * 100)
    elif job.get("status") == "completed":
        progress_percent = 100.0

    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "total_files": total_files,
        "processed_files": processed_files,
        "remaining_files": remaining_files,
        "saved_records": saved_records,
        "progress_percent": progress_percent,
        "eta_seconds": eta_seconds,
        "elapsed_seconds": elapsed_seconds,
        "current_file": job.get("current_file"),
        "error_message": job.get("error_message"),
        "recent_logs": list(job.get("recent_logs") or []),
        "started_at": started_at.isoformat() if started_at else None,
        "last_heartbeat_at": (
            job["last_heartbeat_at"].isoformat() if job.get("last_heartbeat_at") else None
        ),
        "finished_at": finished_at.isoformat() if finished_at else None,
    }


def _parse_allowed_origins() -> list[str]:
    configured = os.environ.get("ALLOWED_ORIGINS", "")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    if origins:
        return origins
    return [
        "http://localhost:3000",
        "https://benverde.vercel.app",
        "https://benverde-v2.vercel.app",
        "https://benverde-v2-arthurcordeiro-coders-projects.vercel.app",
        "https://benverde-v2-git-master-arthurcordeiro-coders-projects.vercel.app",
    ]


allowed_origins = _parse_allowed_origins()


def _normalize_processed_pedidos_payload(payload: ProcessedPedidosPayload) -> tuple[list[dict], list[str]]:
    registros_normalizados: list[dict] = []
    arquivos_presentes: set[str] = set()

    for indice, record in enumerate(payload.registros, start=1):
        registro = record.to_storage_dict()
        if not registro["Produto"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Registro #{indice} sem Produto.",
            )
        if not registro["Loja"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Registro #{indice} sem Loja.",
            )
        if not registro["ARQUIVO"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Registro #{indice} sem ARQUIVO.",
            )
        if registro["QUANT"] < 0 or registro["VALOR TOTAL"] < 0 or registro["VALOR UNIT"] < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Registro #{indice} possui valores negativos.",
            )
        registros_normalizados.append(registro)
        arquivos_presentes.add(registro["ARQUIVO"])

    arquivos_declarados = [Path(nome).name.strip() for nome in payload.arquivos if Path(nome).name.strip()]
    if payload.total_arquivos is not None and payload.total_arquivos < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="total_arquivos nao pode ser negativo.",
        )
    if not registros_normalizados:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Envie ao menos um registro processado.",
        )
    if arquivos_declarados:
        arquivos_declarados_set = set(arquivos_declarados)
        faltantes = sorted(arquivos_presentes - arquivos_declarados_set)
        if faltantes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Arquivos dos registros nao declarados em 'arquivos': {', '.join(faltantes[:10])}",
            )
    else:
        arquivos_declarados = sorted(arquivos_presentes)
    if payload.total_arquivos is not None and payload.total_arquivos != len(arquivos_declarados):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "total_arquivos difere da quantidade de arquivos declarados: "
                f"{payload.total_arquivos} != {len(arquivos_declarados)}."
            ),
        )

    return registros_normalizados, arquivos_declarados


def _normalize_column_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").strip())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_market_name(column_name: object) -> str | None:
    text = str(column_name or "").strip()
    if not text:
        return None

    match = re.search(r"\(([^)]+)\)", text)
    if match:
        return match.group(1).strip()

    normalized = _normalize_column_name(text)
    if "semar" in normalized:
        return "Semar"
    return text


def _coerce_price_value(value: object) -> float | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"nan", "none", "-"}:
        return None

    cleaned = re.sub(r"[^\d,.\-]", "", raw)
    if not cleaned:
        return None

    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")

    try:
        number = float(cleaned)
    except ValueError:
        return None

    if number <= 0:
        return None
    return round(number, 2)


def _canonical_market_name(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None

    normalized = _normalize_column_name(text)
    if "semar" in normalized:
        return "Semar"

    if normalized.isascii():
        return " ".join(part.capitalize() for part in normalized.split())
    return text


def _merge_price_items(raw_items: list[dict], markets: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for item in raw_items:
        grouped.setdefault(item["produto"], []).append(item)

    items: list[dict] = []
    for produto, group in grouped.items():
        if len(group) == 1:
            items.append(group[0])
            continue

        merged_prices: dict[str, float | None] = {}
        for market in markets:
            values = [
                entry["prices"][market]
                for entry in group
                if entry["prices"].get(market) is not None
            ]
            merged_prices[market] = round(sum(values) / len(values), 2) if values else None

        merged_statuses: dict[str, str] = {}
        merged_matches: dict[str, str] = {}
        for entry in group:
            for market in markets:
                if market not in merged_statuses and entry.get("statuses", {}).get(market):
                    merged_statuses[market] = entry["statuses"][market]
                if market not in merged_matches and entry.get("matches", {}).get(market):
                    merged_matches[market] = entry["matches"][market]

        items.append(
            {
                "produto": produto,
                "prices": merged_prices,
                "statuses": merged_statuses,
                "matches": merged_matches,
            }
        )

    items.sort(key=lambda current: current["produto"])
    return items


def _build_price_snapshot_items(df) -> tuple[list[dict], list[str]]:
    if df is None or df.empty:
        return [], []

    columns = list(df.columns)
    product_column = next(
        (
            column
            for column in columns
            if "produto buscado" in _normalize_column_name(column)
        ),
        None,
    )
    if product_column is None:
        product_column = next(
            (column for column in columns if "produto" in _normalize_column_name(column)),
            None,
        )
    if product_column is None:
        return [], []

    market_column = next(
        (
            column
            for column in columns
            if _normalize_column_name(column) in {"estabelecimento", "mercado", "loja", "concorrente"}
        ),
        None,
    )
    generic_price_column = next(
        (
            column
            for column in columns
            if _normalize_column_name(column) in {"preco", "valor", "price"}
        ),
        None,
    )
    generic_status_column = next(
        (
            column
            for column in columns
            if _normalize_column_name(column) == "status"
        ),
        None,
    )
    generic_match_column = next(
        (
            column
            for column in columns
            if "produto encontrado" in _normalize_column_name(column)
        ),
        None,
    )

    if market_column and generic_price_column:
        grouped_rows: dict[str, dict] = {}
        all_markets: set[str] = {"Semar"}

        for row in df.to_dict(orient="records"):
            product = str(row.get(product_column) or "").strip().upper()
            market = _canonical_market_name(row.get(market_column))
            if not product or not market:
                continue

            all_markets.add(market)
            current = grouped_rows.setdefault(
                product,
                {
                    "price_lists": {},
                    "statuses": {},
                    "matches": {},
                },
            )

            price_value = _coerce_price_value(row.get(generic_price_column))
            if price_value is not None:
                current["price_lists"].setdefault(market, []).append(price_value)

            status_value = str(row.get(generic_status_column or "") or "").strip()
            if status_value and market not in current["statuses"]:
                current["statuses"][market] = status_value

            matched_name = str(row.get(generic_match_column or "") or "").strip()
            if matched_name and market not in current["matches"]:
                current["matches"][market] = matched_name

        markets = ["Semar", *sorted(market for market in all_markets if market != "Semar")]
        raw_items = []
        for product, payload in grouped_rows.items():
            prices = {
                market: (
                    round(
                        sum(payload["price_lists"].get(market, []))
                        / len(payload["price_lists"][market]),
                        2,
                    )
                    if payload["price_lists"].get(market)
                    else None
                )
                for market in markets
            }
            if not any(value is not None for value in prices.values()):
                continue
            raw_items.append(
                {
                    "produto": product,
                    "prices": prices,
                    "statuses": payload["statuses"],
                    "matches": payload["matches"],
                }
            )

        return _merge_price_items(raw_items, markets), markets

    price_columns: dict[str, str] = {}
    status_columns: dict[str, str] = {}
    match_columns: dict[str, str] = {}

    for column in columns:
        normalized = _normalize_column_name(column)
        market = _extract_market_name(column)
        if not market:
            continue

        if "preco" in normalized:
            price_columns[market] = column
            continue
        if normalized.startswith("status"):
            status_columns[market] = column
            continue
        if "produto encontrado" in normalized:
            match_columns[market] = column

    if "Semar" not in price_columns:
        orphan_keys = [
            k for k in price_columns
            if _normalize_column_name(k) in {"preco", "preco semar", "price", "valor"}
        ]
        if orphan_keys:
            price_columns["Semar"] = price_columns.pop(orphan_keys[0])

    markets = ["Semar", *sorted(market for market in price_columns if market != "Semar")]
    raw_items: list[dict] = []

    for row in df.to_dict(orient="records"):
        product = str(row.get(product_column) or "").strip().upper()
        if not product:
            continue

        prices: dict[str, float | None] = {}
        statuses: dict[str, str] = {}
        matches: dict[str, str] = {}
        has_any_price = False

        for market in markets:
            price_value = _coerce_price_value(row.get(price_columns.get(market, "")))
            prices[market] = price_value
            if price_value is not None:
                has_any_price = True

            status_value = str(row.get(status_columns.get(market, "")) or "").strip()
            if status_value:
                statuses[market] = status_value

            matched_name = str(row.get(match_columns.get(market, "")) or "").strip()
            if matched_name:
                matches[market] = matched_name

        if not has_any_price:
            continue

        raw_items.append(
            {
                "produto": product,
                "prices": prices,
                "statuses": statuses,
                "matches": matches,
            }
        )

    return _merge_price_items(raw_items, markets), markets


def _build_price_overview(pasta_precos: Path) -> dict:
    datasets = load_precos(str(pasta_precos))
    if not datasets:
        return {"latestDate": None, "dates": [], "markets": ["Semar"], "snapshots": {}}

    dates: list[dict] = []
    snapshots: dict[str, list[dict]] = {}
    all_markets: set[str] = {"Semar"}

    for date_key, df in datasets.items():
        items, markets = _build_price_snapshot_items(df)
        snapshots[date_key] = items
        all_markets.update(markets)

        try:
            label = datetime.strptime(date_key, "%d-%m-%Y").strftime("%d/%m/%Y")
        except ValueError:
            label = date_key

        dates.append({"key": date_key, "label": label})

    ordered_markets = ["Semar", *sorted(market for market in all_markets if market != "Semar")]
    return {
        "latestDate": dates[0]["key"] if dates else None,
        "dates": dates,
        "markets": ordered_markets,
        "snapshots": snapshots,
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


@app.options("/{full_path:path}", include_in_schema=False)
def preflight_handler(full_path: str) -> Response:
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api/estoque/saldo")
def get_estoque_saldo(current_user: dict = Depends(get_current_user)):
    pasta_entradas = Path(__file__).resolve().parent / "dados" / "entradas_bananas"
    pasta_saidas = Path(__file__).resolve().parent / "dados" / "saidas_bananas"
    saldo, historico = calcular_estoque(
        pasta_entradas=str(pasta_entradas),
        pasta_saidas=str(pasta_saidas),
    )
    return {
        "saldo": saldo,
        "historico": jsonable_encoder(historico),
    }


@app.get("/api/estoque/movimentacoes")
def get_movimentacoes(current_user: dict = Depends(get_current_user)):
    movimentacoes = load_movimentacoes_manuais(caminho_json="")
    return jsonable_encoder(movimentacoes)


@app.post("/api/estoque/movimentacao")
def post_movimentacao(
    payload: dict | list = Body(...), current_user: dict = Depends(get_current_user)
):
    if isinstance(payload, dict):
        registros = [payload]
    elif isinstance(payload, list):
        registros = payload
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload deve ser objeto ou lista de objetos.",
        )

    salvar_movimentacao_manual(registros=registros, caminho_json="")
    return {"success": True, "saved": len(registros)}


@app.delete("/api/estoque/movimentacao/{id}")
def delete_movimentacao(id: int, current_user: dict = Depends(get_current_user)):
    deletar_movimentacao_manual(entry_id=id, caminho_json="")
    return {"success": True, "deleted_id": id}


@app.get("/api/caixas")
def get_caixas(current_user: dict = Depends(get_current_user)):
    df = load_registros_caixas()
    return jsonable_encoder(df.to_dict(orient="records"))


@app.post("/api/caixas")
def post_caixas(payload: dict, current_user: dict = Depends(get_current_user)):
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload deve ser um objeto JSON.",
        )
    salvar_registro_caixas(payload)
    return {"success": True}


@app.get("/api/precos")
def get_precos(current_user: dict = Depends(get_current_user)):
    pasta_precos = Path(__file__).resolve().parent / "dados" / "precos"
    precos = listar_precos_consolidados(str(pasta_precos))
    return jsonable_encoder(precos)


@app.get("/api/precos/overview")
def get_precos_overview(current_user: dict = Depends(get_current_user)):
    pasta_precos = Path(__file__).resolve().parent / "dados" / "precos"
    overview = _build_price_overview(pasta_precos)
    return jsonable_encoder(overview)


@app.post("/api/mita-ai/chat")
def post_mita_ai_chat(
    payload: MitaChatPayload, current_user: dict = Depends(get_current_user)
):
    return chat_with_mita(message=payload.message, history=payload.history)


@app.post("/api/upload/pedidos", status_code=status.HTTP_202_ACCEPTED)
async def upload_pedidos(
    files: list[UploadFile] = File(...), current_user: dict = Depends(get_current_user)
):
    is_admin = bool(
        current_user.get("role") == "admin" or current_user.get("is_admin") is True
    )
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Apenas administradores podem importar pedidos.",
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Envie ao menos um arquivo PDF ou ZIP.",
        )

    try:
        job_id = uuid.uuid4().hex
        job_dir = IMPORT_JOBS_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        pdf_paths: list[str] = []

        for file in files:
            filename = Path(file.filename or "").name
            if not filename:
                continue

            lower_name = filename.lower()
            if not lower_name.endswith((".pdf", ".zip")):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Formato nao suportado: {filename}",
                )

            staged_path = job_dir / f"{uuid.uuid4().hex}_{filename}"
            with open(staged_path, "wb") as out_file:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    out_file.write(chunk)

            if lower_name.endswith(".pdf"):
                pdf_paths.append(str(staged_path))
                continue

            try:
                with zipfile.ZipFile(staged_path) as zip_file:
                    for member in zip_file.infolist():
                        if member.is_dir():
                            continue

                        member_name = Path(member.filename).name
                        if not member_name.lower().endswith(".pdf"):
                            continue

                        extracted_path = job_dir / f"{uuid.uuid4().hex}_{member_name}"
                        with zip_file.open(member) as source, open(
                            extracted_path, "wb"
                        ) as target:
                            shutil.copyfileobj(source, target)
                        pdf_paths.append(str(extracted_path))
            except zipfile.BadZipFile as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Arquivo ZIP invalido: {filename}",
                ) from exc
            finally:
                if staged_path.exists():
                    staged_path.unlink(missing_ok=True)

        if not pdf_paths:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nenhum PDF valido encontrado para processamento.",
            )

        create_import_job(
            job_id,
            total_files=len(pdf_paths),
            recent_logs=[f"Upload recebido com {len(pdf_paths)} arquivo(s) para processamento."],
        )

        process = Process(
            target=run_import_job,
            args=(job_id, pdf_paths, str(job_dir)),
            daemon=True,
        )
        try:
            process.start()
        except Exception:
            update_import_job(
                job_id,
                status="failed",
                error_message="Nao foi possivel iniciar o processo em background.",
                touch_heartbeat=True,
                finished=True,
            )
            raise

        job = get_import_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Nao foi possivel iniciar o job de importacao.",
            )
        return _serialize_import_job(job)
    except HTTPException:
        if "job_dir" in locals() and job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        logger.exception("Falha ao processar upload de pedidos")
        if "job_dir" in locals() and job_dir.exists():
            shutil.rmtree(job_dir, ignore_errors=True)
        detail = str(exc).strip() or "Falha interna ao processar upload de pedidos."
        raise HTTPException(status_code=500, detail=detail)
    finally:
        for file in files:
            await file.close()


@app.get("/api/upload/pedidos/{job_id}")
def get_upload_pedidos_status(job_id: str, current_user: dict = Depends(get_current_user)):
    is_admin = bool(
        current_user.get("role") == "admin" or current_user.get("is_admin") is True
    )
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Apenas administradores podem consultar importacoes de pedidos.",
        )

    job = get_import_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Importacao nao encontrada.",
        )

    return _serialize_import_job(job)


@app.post("/api/upload/pedidos-processados", status_code=status.HTTP_202_ACCEPTED)
def upload_pedidos_processados(
    payload: ProcessedPedidosPayload, current_user: dict = Depends(get_current_user)
):
    is_admin = bool(
        current_user.get("role") == "admin" or current_user.get("is_admin") is True
    )
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Apenas administradores podem importar pedidos processados.",
        )

    registros_normalizados, arquivos = _normalize_processed_pedidos_payload(payload)
    job_id = uuid.uuid4().hex

    create_import_job(
        job_id,
        total_files=len(arquivos),
        recent_logs=[
            (
                f"Importacao externa recebida da origem '{payload.origem}' com "
                f"{len(arquivos)} arquivo(s) e {len(registros_normalizados)} registro(s)."
            )
        ],
    )

    try:
        update_import_job(
            job_id,
            status="processing",
            processed_files=0,
            saved_records=0,
            current_file=None,
            error_message=None,
            touch_heartbeat=True,
        )
        replace_cache_pedidos(registros_normalizados)
        update_import_job(
            job_id,
            status="completed",
            processed_files=len(arquivos),
            saved_records=len(registros_normalizados),
            current_file=None,
            error_message=None,
            recent_logs=[
                (
                    f"Importacao externa concluida da origem '{payload.origem}' com "
                    f"{len(arquivos)} arquivo(s) e {len(registros_normalizados)} registro(s)."
                )
            ],
            touch_heartbeat=True,
            finished=True,
        )
    except Exception as exc:
        logger.exception("Falha ao salvar pedidos processados recebidos externamente")
        update_import_job(
            job_id,
            status="failed",
            processed_files=0,
            saved_records=0,
            current_file=None,
            error_message=str(exc).strip() or "Falha ao salvar pedidos processados.",
            touch_heartbeat=True,
            finished=True,
        )
        detail = str(exc).strip() or "Falha interna ao salvar pedidos processados."
        raise HTTPException(status_code=500, detail=detail) from exc

    job = get_import_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Nao foi possivel recuperar o status da importacao processada.",
        )
    response = _serialize_import_job(job)
    response["source"] = payload.origem
    response["files"] = arquivos
    return response


@app.post("/api/upload/pdf")
async def upload_pdf(
    file: UploadFile = File(...), current_user: dict = Depends(get_current_user)
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Envie um arquivo PDF valido.",
        )

    temp_dir = Path(__file__).resolve().parent / "temp_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4().hex}_{Path(file.filename).name}"

    try:
        with open(temp_path, "wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)

        resultado = extrair_bananas_pdf_upload(str(temp_path))
        return {"arquivo": file.filename, "resultado": resultado}
    finally:
        await file.close()
        if temp_path.exists():
            os.remove(temp_path)
