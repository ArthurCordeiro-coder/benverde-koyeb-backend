"""auth.py
Sistema de autenticação para o app Benverde.
Usa apenas stdlib: hashlib, secrets, json, threading, datetime, re, os.
"""

import hashlib
import re
import secrets
import threading
from datetime import datetime, timedelta, timezone

from db import get_connection

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Internos
# ---------------------------------------------------------------------------

def _hash_senha(salt: str, senha: str) -> str:
    return hashlib.sha256((salt + senha).encode()).hexdigest()


def _normalizar_role(role: str | None, is_admin: bool | None = False) -> str:
    if role in {"admin", "operacional"}:
        return role
    return "admin" if is_admin else "operacional"


# ---------------------------------------------------------------------------
# Leitura / escrita pública
# ---------------------------------------------------------------------------

def carregar_users() -> list[dict]:
    with _LOCK:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT username, nome, salt, senha_hash, is_admin, criado_em, funcionalidade, role
                    FROM users ORDER BY username
                    """
                )
                return [
                    {
                        "username": row[0],
                        "nome": row[1],
                        "salt": row[2],
                        "senha_hash": row[3],
                        "is_admin": row[4],
                        "criado_em": row[5],
                        "funcionalidade": row[6],
                        "role": _normalizar_role(row[7], row[4]),
                    }
                    for row in cur.fetchall()
                ]


def salvar_users(users: list[dict]) -> None:
    with _LOCK:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM users")
                for user in users:
                    role = _normalizar_role(
                        user.get("role"), bool(user.get("is_admin", False))
                    )
                    cur.execute(
                        """
                        INSERT INTO users (
                            username, nome, salt, senha_hash, is_admin, criado_em, funcionalidade, role
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            user.get("username"),
                            user.get("nome"),
                            user.get("salt"),
                            user.get("senha_hash"),
                            role == "admin",
                            user.get("criado_em"),
                            user.get("funcionalidade", "administracao geral"),
                            role,
                        ),
                    )


def carregar_pending() -> list[dict]:
    with _LOCK:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT username, nome, salt, senha_hash, solicitado_em, funcionalidade
                    FROM pending ORDER BY username
                    """
                )
                return [
                    {
                        "username": row[0],
                        "nome": row[1],
                        "salt": row[2],
                        "senha_hash": row[3],
                        "solicitado_em": row[4],
                        "funcionalidade": row[5],
                    }
                    for row in cur.fetchall()
                ]


def salvar_pending(pending: list[dict]) -> None:
    with _LOCK:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM pending")
                for entry in pending:
                    cur.execute(
                        """
                        INSERT INTO pending (
                            username, nome, salt, senha_hash, solicitado_em, funcionalidade
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            entry.get("username"),
                            entry.get("nome"),
                            entry.get("salt"),
                            entry.get("senha_hash"),
                            entry.get("solicitado_em"),
                            entry.get("funcionalidade", "administracao geral"),
                        ),
                    )


def carregar_lockouts() -> dict:
    with _LOCK:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT username, tentativas, bloqueado_ate
                    FROM lockouts
                    """
                )
                return {
                    row[0]: {"tentativas": row[1], "bloqueado_ate": row[2]}
                    for row in cur.fetchall()
                }


def salvar_lockouts(lockouts: dict) -> None:
    with _LOCK:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM lockouts")
                for username, entry in lockouts.items():
                    cur.execute(
                        """
                        INSERT INTO lockouts (username, tentativas, bloqueado_ate)
                        VALUES (%s, %s, %s)
                        """,
                        (username, entry.get("tentativas", 0), entry.get("bloqueado_ate")),
                    )


def get_user(username: str) -> dict | None:
    """Busca um usuário aprovado pelo username."""
    user = next((u for u in carregar_users() if u["username"] == username), None)
    if user:
        user.setdefault("funcionalidade", "administracao geral")
        user["role"] = _normalizar_role(
            user.get("role"), bool(user.get("is_admin", False))
        )
        user["is_admin"] = user["role"] == "admin"
    return user


# ---------------------------------------------------------------------------
# Lógica de autenticação
# ---------------------------------------------------------------------------

def verificar_login(username: str, senha: str) -> tuple[bool, str]:
    """
    Valida credenciais com proteção contra brute-force.
    Retorna (True, "ok") ou (False, "motivo").
    """
    with _LOCK:
        agora = datetime.now(timezone.utc)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT tentativas, bloqueado_ate FROM lockouts WHERE username = %s",
                    (username,),
                )
                row = cur.fetchone()
                tentativas = row[0] if row else 0
                bloqueado_ate = row[1] if row else None
                if bloqueado_ate and isinstance(bloqueado_ate, str):
                    bloqueado_ate = datetime.fromisoformat(bloqueado_ate)

                if bloqueado_ate and agora < bloqueado_ate:
                    hora = bloqueado_ate.astimezone().strftime("%H:%M")
                    return False, f"Usuário bloqueado até {hora}"

                def _registrar_tentativa() -> tuple[bool, str]:
                    nonlocal tentativas
                    tentativas = (tentativas or 0) + 1
                    bloqueado_destino = None
                    if tentativas >= 5:
                        bloqueado_destino = agora + timedelta(minutes=15)
                        tentativas = 0
                    cur.execute(
                        """
                        INSERT INTO lockouts (username, tentativas, bloqueado_ate)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (username) DO UPDATE
                          SET tentativas = EXCLUDED.tentativas,
                              bloqueado_ate = EXCLUDED.bloqueado_ate
                        """,
                        (username, tentativas, bloqueado_destino),
                    )
                    if bloqueado_destino:
                        return False, "Muitas tentativas. Usuário bloqueado por 15 minutos."
                    return False, f"Usuário ou senha inválidos ({tentativas} de 5)"

                cur.execute(
                    """
                    SELECT username, salt, senha_hash, funcionalidade
                    FROM users WHERE username = %s
                    """,
                    (username,),
                )
                user_row = cur.fetchone()

                if user_row is None:
                    return _registrar_tentativa()

                salt = user_row[1]
                senha_hash = user_row[2]
                if _hash_senha(salt, senha) != senha_hash:
                    return _registrar_tentativa()

                # Zera lockout após sucesso
                cur.execute(
                    """
                    INSERT INTO lockouts (username, tentativas, bloqueado_ate)
                    VALUES (%s, 0, NULL)
                    ON CONFLICT (username) DO UPDATE
                      SET tentativas = EXCLUDED.tentativas,
                          bloqueado_ate = EXCLUDED.bloqueado_ate
                    """,
                    (username,),
                )
                return True, "ok"


def registrar_usuario(
    username: str,
    nome: str,
    senha: str,
    funcionalidade: str = "administracao geral",
) -> tuple[bool, str]:
    """
    Registra um novo usuário.
    Retorna (True, "admin_criado") | (True, "pendente") | (False, "motivo").
    """
    if not re.fullmatch(r"[a-zA-Z0-9_]{3,20}", username):
        return False, "Username deve ter 3–20 caracteres (letras, números e _)"
    if len(senha) < 6:
        return False, "Senha deve ter pelo menos 6 caracteres"

    with _LOCK:
        salt = secrets.token_hex(32)
        hash_ = _hash_senha(salt, senha)
        agora_iso = datetime.now(timezone.utc).isoformat()
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM users WHERE username = %s", (username,))
                if cur.fetchone():
                    return False, "Username já cadastrado"
                cur.execute("SELECT 1 FROM pending WHERE username = %s", (username,))
                if cur.fetchone():
                    return False, "Username já aguarda aprovação"
                cur.execute("SELECT COUNT(*) FROM users")
                total_users = cur.fetchone()[0]
                if total_users == 0:
                    cur.execute(
                        """
                        INSERT INTO users (
                            username, nome, salt, senha_hash, is_admin, criado_em, funcionalidade, role
                        ) VALUES (%s, %s, %s, %s, TRUE, %s, %s, 'admin')
                        """,
                        (
                            username,
                            nome,
                            salt,
                            hash_,
                            agora_iso,
                            funcionalidade,
                        ),
                    )
                    return True, "admin_criado"
                cur.execute(
                    """
                    INSERT INTO pending (
                        username, nome, salt, senha_hash, solicitado_em, funcionalidade
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        username,
                        nome,
                        salt,
                        hash_,
                        agora_iso,
                        funcionalidade,
                    ),
                )
                return True, "pendente"


def aprovar_usuario(username: str) -> bool:
    """Move o usuário pendente para users com is_admin=False."""
    with _LOCK:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT username, nome, salt, senha_hash, funcionalidade
                    FROM pending WHERE username = %s
                    """,
                    (username,),
                )
                entry = cur.fetchone()
                if entry is None:
                    return False
                cur.execute(
                    """
                    INSERT INTO users (
                        username, nome, salt, senha_hash, is_admin, criado_em, funcionalidade, role
                    ) VALUES (%s, %s, %s, %s, FALSE, %s, %s, 'operacional')
                    """,
                    (
                        entry[0],
                        entry[1],
                        entry[2],
                        entry[3],
                        datetime.now(timezone.utc).isoformat(),
                        entry[4] or "administracao geral",
                    ),
                )
                cur.execute("DELETE FROM pending WHERE username = %s", (username,))
                return True


def rejeitar_usuario(username: str) -> bool:
    """Remove o usuário de pending."""
    with _LOCK:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM pending WHERE username = %s", (username,))
                return cur.rowcount > 0
