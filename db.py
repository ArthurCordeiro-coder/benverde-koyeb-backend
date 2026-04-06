import json
import logging
import os
import threading
from contextlib import contextmanager

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)
_CONFIG_LOCK = threading.Lock()
_DB_CONFIG = None
_UNSET = object()


def _get_config() -> dict:
    global _DB_CONFIG
    if _DB_CONFIG is not None:
        return _DB_CONFIG

    def _resolve(key: str):
        return os.environ.get(key)

    config = {
        "host": _resolve("PGHOST"),
        "dbname": _resolve("PGDATABASE"),
        "user": _resolve("PGUSER"),
        "password": _resolve("PGPASSWORD"),
        "sslmode": _resolve("PGSSLMODE"),
    }
    port = _resolve("PGPORT")
    if port:
        config["port"] = port
    channel_binding = _resolve("PGCHANNELBINDING")
    if channel_binding:
        config["channel_binding"] = channel_binding

    config = {k: v for k, v in config.items() if v}
    if "port" not in config:
        config["port"] = 5432

    with _CONFIG_LOCK:
        if _DB_CONFIG is None:
            _DB_CONFIG = config
        return _DB_CONFIG


@contextmanager
def get_connection():
    database_url = (os.environ.get("DATABASE_URL") or "").strip()
    cfg = _get_config()
    if not database_url and not cfg:
        raise RuntimeError(
            "Banco de dados nao configurado (falta DATABASE_URL ou PGHOST/PGUSER/PGPASSWORD/PGDATABASE)"
        )
    conn = psycopg.connect(database_url) if database_url else psycopg.connect(**cfg)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _format_identifier(name: str):
    return sql.Identifier(name)


def _format_qualified_identifier(name: str):
    parts = [part.strip() for part in str(name or "").split(".") if part.strip()]
    if not parts:
        raise ValueError("Identificador de tabela vazio.")
    return sql.SQL(".").join(sql.Identifier(part) for part in parts)


def _ensure_tables():
    statements = [
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            nome TEXT,
            salt TEXT,
            senha_hash TEXT,
            is_admin BOOLEAN,
            criado_em TIMESTAMPTZ,
            funcionalidade TEXT,
            role TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS pending (
            username TEXT PRIMARY KEY,
            nome TEXT,
            salt TEXT,
            senha_hash TEXT,
            solicitado_em TIMESTAMPTZ,
            funcionalidade TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS lockouts (
            username TEXT PRIMARY KEY,
            tentativas INTEGER,
            bloqueado_ate TIMESTAMPTZ
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS metas_local (
            produto TEXT PRIMARY KEY,
            meta BIGINT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cache_estoque (
            key TEXT PRIMARY KEY,
            payload JSONB,
            updated_at TIMESTAMPTZ DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cache_pedidos (
            arquivo_pdf TEXT NOT NULL,
            data TIMESTAMPTZ NOT NULL,
            loja TEXT NOT NULL,
            produto TEXT NOT NULL,
            unidade TEXT NOT NULL,
            quant DOUBLE PRECISION NOT NULL,
            valor_total DOUBLE PRECISION NOT NULL,
            valor_unit DOUBLE PRECISION NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS import_jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            total_files INTEGER NOT NULL DEFAULT 0,
            processed_files INTEGER NOT NULL DEFAULT 0,
            saved_records INTEGER NOT NULL DEFAULT 0,
            started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at TIMESTAMPTZ,
            last_heartbeat_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            current_file TEXT,
            error_message TEXT,
            recent_logs JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS estoque_manual (
            id SERIAL PRIMARY KEY,
            data TIMESTAMPTZ,
            tipo TEXT,
            produto TEXT,
            quant DOUBLE PRECISION,
            unidade TEXT,
            loja TEXT,
            arquivo TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS caixas_lojas (
            id SERIAL PRIMARY KEY,
            data DATE,
            loja TEXT,
            n_loja INTEGER,
            caixas_benverde INTEGER,
            caixas_ccj INTEGER,
            ccj_banca INTEGER,
            ccj_mercadoria INTEGER,
            ccj_retirada INTEGER,
            caixas_bananas INTEGER,
            total INTEGER,
            entregue TEXT
        )
        """,
    ]
    with get_connection() as conn:
        with conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)


_JSON_CACHE_TABLES = {"cache_estoque"}
_RELATIONAL_CACHE_TABLES = {"cache_pedidos"}
_CACHE_TABLES = _JSON_CACHE_TABLES | _RELATIONAL_CACHE_TABLES


def _ensure_cache_columns():
    cache_columns = {
        "key": "TEXT",
        "payload": "JSONB",
        "updated_at": "TIMESTAMPTZ DEFAULT now()",
    }
    for table in _JSON_CACHE_TABLES:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for column, definition in cache_columns.items():
                    cur.execute(
                        sql.SQL(
                            "ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {definition}"
                        ).format(
                            table=_format_identifier(table),
                            column=_format_identifier(column),
                            definition=sql.SQL(definition),
                        )
                    )
                cur.execute(
                    sql.SQL(
                        "CREATE UNIQUE INDEX IF NOT EXISTS {idx} ON {table} (key)"
                    ).format(
                        idx=_format_identifier(f"{table}_key_idx"),
                        table=_format_identifier(table),
                    )
                )


def _ensure_import_jobs_columns():
    columns = {
        "job_id": "TEXT",
        "status": "TEXT NOT NULL DEFAULT 'queued'",
        "total_files": "INTEGER NOT NULL DEFAULT 0",
        "processed_files": "INTEGER NOT NULL DEFAULT 0",
        "saved_records": "INTEGER NOT NULL DEFAULT 0",
        "started_at": "TIMESTAMPTZ NOT NULL DEFAULT now()",
        "finished_at": "TIMESTAMPTZ",
        "last_heartbeat_at": "TIMESTAMPTZ NOT NULL DEFAULT now()",
        "current_file": "TEXT",
        "error_message": "TEXT",
        "recent_logs": "JSONB NOT NULL DEFAULT '[]'::jsonb",
        "created_at": "TIMESTAMPTZ NOT NULL DEFAULT now()",
        "updated_at": "TIMESTAMPTZ NOT NULL DEFAULT now()",
    }
    with get_connection() as conn:
        with conn.cursor() as cur:
            for column, definition in columns.items():
                cur.execute(
                    sql.SQL(
                        "ALTER TABLE import_jobs ADD COLUMN IF NOT EXISTS {column} {definition}"
                    ).format(
                        column=_format_identifier(column),
                        definition=sql.SQL(definition),
                    )
                )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS import_jobs_job_id_idx
                ON import_jobs (job_id)
                """
            )


def _ensure_cache_pedidos_columns():
    columns = {
        "arquivo_pdf": "TEXT",
        "data": "TIMESTAMPTZ",
        "loja": "TEXT",
        "produto": "TEXT",
        "unidade": "TEXT",
        "quant": "DOUBLE PRECISION",
        "valor_total": "DOUBLE PRECISION",
        "valor_unit": "DOUBLE PRECISION",
        "updated_at": "TIMESTAMPTZ DEFAULT now()",
    }
    with get_connection() as conn:
        with conn.cursor() as cur:
            for column, definition in columns.items():
                cur.execute(
                    sql.SQL(
                        "ALTER TABLE cache_pedidos ADD COLUMN IF NOT EXISTS {column} {definition}"
                    ).format(
                        column=_format_identifier(column),
                        definition=sql.SQL(definition),
                    )
                )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS cache_pedidos_arquivo_pdf_idx
                ON cache_pedidos (arquivo_pdf)
                """
            )


def _ensure_auth_columns():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT")
            cur.execute(
                """
                UPDATE users
                SET role = CASE
                    WHEN role IN ('admin', 'operacional') THEN role
                    WHEN COALESCE(is_admin, FALSE) THEN 'admin'
                    ELSE 'operacional'
                END
                WHERE role IS NULL OR role NOT IN ('admin', 'operacional')
                """
            )
            cur.execute("ALTER TABLE users ALTER COLUMN role SET DEFAULT 'operacional'")
            cur.execute("ALTER TABLE users ALTER COLUMN role SET NOT NULL")


def _ensure_db_structures():
    _ensure_tables()
    _ensure_auth_columns()
    _ensure_cache_columns()
    _ensure_cache_pedidos_columns()
    _ensure_import_jobs_columns()


_ensure_db_structures()


def fetch_cache(table_name: str) -> dict:
    if table_name not in _CACHE_TABLES:
        return {}
    if table_name == "cache_pedidos":
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT arquivo_pdf, data, loja, produto, unidade, quant, valor_total, valor_unit
                    FROM cache_pedidos
                    WHERE arquivo_pdf IS NOT NULL
                    ORDER BY data DESC NULLS LAST, arquivo_pdf, produto
                    """
                )
                cache: dict = {}
                for row in cur.fetchall():
                    arquivo_pdf = str(row[0] or "").strip()
                    if not arquivo_pdf:
                        continue
                    cache.setdefault(arquivo_pdf, []).append(
                        {
                            "data": row[1].isoformat() if row[1] else None,
                            "loja": row[2],
                            "produto": row[3],
                            "unidade": row[4],
                            "quant": float(row[5] or 0),
                            "valor_total": float(row[6] or 0),
                            "valor_unit": float(row[7] or 0),
                        }
                    )
                return cache
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT key, payload FROM {table} WHERE key IS NOT NULL AND payload IS NOT NULL"
                ).format(table=_format_identifier(table_name))
            )
            cache = {}
            for key, payload in cur.fetchall():
                if isinstance(payload, str):
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        pass
                cache[key] = payload
            return cache


def save_cache_pedidos_relacional(records: list[dict]) -> None:
    if not records:
        return

    valores = []
    arquivos = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        arquivo_pdf = str(record.get("arquivo_pdf") or "").strip()
        if not arquivo_pdf:
            continue
        arquivos.add(arquivo_pdf)
        valores.append(
            (
                arquivo_pdf,
                record.get("data"),
                str(record.get("loja") or "").strip(),
                str(record.get("produto") or "").strip(),
                str(record.get("unidade") or "KG").strip() or "KG",
                float(record.get("quant") or 0.0),
                float(record.get("valor_total") or 0.0),
                float(record.get("valor_unit") or 0.0),
            )
        )

    if not valores:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            if arquivos:
                cur.execute(
                    "DELETE FROM cache_pedidos WHERE arquivo_pdf = ANY(%s)",
                    (list(arquivos),),
                )
            cur.executemany(
                """
                INSERT INTO cache_pedidos
                (arquivo_pdf, data, loja, produto, unidade, quant, valor_total, valor_unit, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                """,
                valores,
            )


def replace_cache_pedidos(records: list[dict]) -> None:
    normalizados = []
    for record in records:
        if not isinstance(record, dict):
            continue

        arquivo_pdf = str(record.get("arquivo_pdf") or record.get("ARQUIVO") or "").strip()
        if not arquivo_pdf:
            continue
        if not arquivo_pdf.startswith(("pedido::", "semar::")):
            arquivo_pdf = f"pedido::{arquivo_pdf}"

        normalizados.append(
            {
                "arquivo_pdf": arquivo_pdf,
                "data": record.get("data", record.get("Data")),
                "loja": record.get("loja", record.get("Loja")),
                "produto": record.get("produto", record.get("Produto")),
                "unidade": record.get("unidade", record.get("UNID", "KG")),
                "quant": record.get("quant", record.get("QUANT", 0)),
                "valor_total": record.get("valor_total", record.get("VALOR TOTAL", 0)),
                "valor_unit": record.get("valor_unit", record.get("VALOR UNIT", 0)),
            }
        )

    save_cache_pedidos_relacional(normalizados)


def upsert_cache(table_name: str, entries: dict) -> None:
    if table_name not in _CACHE_TABLES or not entries:
        return
    if table_name == "cache_pedidos":
        registros = []
        for arquivo_pdf, itens in entries.items():
            if not isinstance(itens, list):
                continue
            for item in itens:
                if not isinstance(item, dict):
                    continue
                registros.append(
                    {
                        "arquivo_pdf": arquivo_pdf,
                        "data": item.get("data"),
                        "loja": item.get("loja"),
                        "produto": item.get("produto"),
                        "unidade": item.get("unidade", "KG"),
                        "quant": item.get("quant", 0),
                        "valor_total": item.get("valor_total", 0),
                        "valor_unit": item.get("valor_unit", 0),
                    }
                )
        save_cache_pedidos_relacional(registros)
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            stmt = sql.SQL(
                """
                INSERT INTO {table} (key, payload, updated_at)
                VALUES (%s, %s, now())
                ON CONFLICT (key) DO UPDATE
                  SET payload = EXCLUDED.payload,
                      updated_at = EXCLUDED.updated_at
                """
            ).format(table=_format_identifier(table_name))
            for key, payload in entries.items():
                payload_json = (
                    payload
                    if isinstance(payload, str)
                    else json.dumps(payload, ensure_ascii=False, default=str)
                )
                cur.execute(stmt, (key, payload_json))


def clear_cache_table(table_name: str) -> None:
    if table_name not in _CACHE_TABLES:
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {table}").format(table=_format_identifier(table_name))
            )


def fetch_precos_rows() -> list[dict]:
    table_name = os.environ.get("PRECOS_TABLE", "precos").strip() or "precos"

    try:
        qualified_table = _format_qualified_identifier(table_name)
    except ValueError:
        logger.warning("PRECOS_TABLE invalida: %s", table_name)
        return []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (table_name,))
            if cur.fetchone()[0] is None:
                logger.info("Tabela de precos nao encontrada no banco: %s", table_name)
                return []

            cur.execute(sql.SQL("SELECT * FROM {table}").format(table=qualified_table))
            columns = [
                item.name if hasattr(item, "name") else item[0]
                for item in (cur.description or [])
            ]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def load_metas() -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT produto, meta FROM metas_local ORDER BY produto")
            return [{"Produto": row[0], "Meta": row[1]} for row in cur.fetchall()]


def replace_metas(records: list[dict]) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM metas_local")
            for item in records:
                cur.execute(
                    "INSERT INTO metas_local (produto, meta) VALUES (%s, %s)",
                    (item.get("Produto"), item.get("Meta")),
                )


def insert_movimentacoes(records: list[dict]) -> None:
    if not records:
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            for rec in records:
                cur.execute(
                    """
                    INSERT INTO estoque_manual (data, tipo, produto, quant, unidade, loja, arquivo)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        rec.get("data"),
                        rec.get("tipo"),
                        rec.get("produto"),
                        rec.get("quant"),
                        rec.get("unidade"),
                        rec.get("loja"),
                        rec.get("arquivo"),
                    ),
                )


def fetch_movimentacoes() -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, data, tipo, produto, quant, unidade, loja, arquivo
                FROM estoque_manual
                ORDER BY id
                """
            )
            return [
                {
                    "id": row[0],
                    "data": row[1],
                    "tipo": row[2],
                    "produto": row[3],
                    "quant": row[4],
                    "unidade": row[5],
                    "loja": row[6],
                    "arquivo": row[7],
                }
                for row in cur.fetchall()
            ]


def delete_movimentacao(entry_id: int) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM estoque_manual WHERE id = %s", (entry_id,))


def insert_caixa(registro: dict) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO caixas_lojas (
                    data, loja, n_loja,
                    caixas_benverde, caixas_ccj, ccj_banca,
                    ccj_mercadoria, ccj_retirada, caixas_bananas,
                    total, entregue
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    registro.get("data"),
                    registro.get("loja"),
                    registro.get("n_loja"),
                    registro.get("caixas_benverde"),
                    registro.get("caixas_ccj"),
                    registro.get("ccj_banca"),
                    registro.get("ccj_mercadoria"),
                    registro.get("ccj_retirada"),
                    registro.get("caixas_bananas"),
                    registro.get("total"),
                    registro.get("entregue"),
                ),
            )


def fetch_caixas() -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT data, loja, n_loja, caixas_benverde, caixas_ccj,
                       ccj_banca, ccj_mercadoria, ccj_retirada,
                       caixas_bananas, total, entregue
                FROM caixas_lojas
                ORDER BY data DESC NULLS LAST, loja, n_loja
                """
            )
            return [
                {
                    "data": row[0],
                    "loja": row[1],
                    "n_loja": row[2],
                    "caixas_benverde": row[3],
                    "caixas_ccj": row[4],
                    "ccj_banca": row[5],
                    "ccj_mercadoria": row[6],
                    "ccj_retirada": row[7],
                    "caixas_bananas": row[8],
                    "total": row[9],
                    "entregue": row[10],
                }
                for row in cur.fetchall()
            ]


def create_import_job(
    job_id: str, total_files: int, recent_logs: list[str] | None = None
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO import_jobs (
                    job_id,
                    status,
                    total_files,
                    processed_files,
                    saved_records,
                    started_at,
                    last_heartbeat_at,
                    recent_logs,
                    created_at,
                    updated_at
                )
                VALUES (%s, 'queued', %s, 0, 0, now(), now(), %s, now(), now())
                """,
                (job_id, max(0, int(total_files)), Jsonb(recent_logs or [])),
            )


def update_import_job(
    job_id: str,
    *,
    status: str | None = None,
    total_files: int | None = None,
    processed_files: int | None = None,
    saved_records: int | None = None,
    current_file=_UNSET,
    error_message=_UNSET,
    recent_logs=_UNSET,
    finished: bool = False,
    touch_heartbeat: bool = False,
) -> None:
    assignments: list[sql.SQL] = []
    values: list[object] = []

    if status is not None:
        assignments.append(sql.SQL("status = %s"))
        values.append(status)
    if total_files is not None:
        assignments.append(sql.SQL("total_files = %s"))
        values.append(max(0, int(total_files)))
    if processed_files is not None:
        assignments.append(sql.SQL("processed_files = %s"))
        values.append(max(0, int(processed_files)))
    if saved_records is not None:
        assignments.append(sql.SQL("saved_records = %s"))
        values.append(max(0, int(saved_records)))
    if current_file is not _UNSET:
        assignments.append(sql.SQL("current_file = %s"))
        values.append(current_file)
    if error_message is not _UNSET:
        assignments.append(sql.SQL("error_message = %s"))
        values.append(error_message)
    if recent_logs is not _UNSET:
        assignments.append(sql.SQL("recent_logs = %s"))
        values.append(Jsonb(recent_logs))
    if touch_heartbeat:
        assignments.append(sql.SQL("last_heartbeat_at = now()"))
    if finished:
        assignments.append(sql.SQL("finished_at = now()"))

    assignments.append(sql.SQL("updated_at = now()"))
    values.append(job_id)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("UPDATE import_jobs SET {assignments} WHERE job_id = %s").format(
                    assignments=sql.SQL(", ").join(assignments),
                ),
                values,
            )


def get_import_job(job_id: str) -> dict | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    job_id,
                    status,
                    total_files,
                    processed_files,
                    saved_records,
                    started_at,
                    finished_at,
                    last_heartbeat_at,
                    current_file,
                    error_message,
                    recent_logs
                FROM import_jobs
                WHERE job_id = %s
                """,
                (job_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return {
                "job_id": row[0],
                "status": row[1],
                "total_files": int(row[2] or 0),
                "processed_files": int(row[3] or 0),
                "saved_records": int(row[4] or 0),
                "started_at": row[5],
                "finished_at": row[6],
                "last_heartbeat_at": row[7],
                "current_file": row[8],
                "error_message": row[9],
                "recent_logs": list(row[10] or []),
            }
