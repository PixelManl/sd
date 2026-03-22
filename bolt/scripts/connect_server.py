from __future__ import annotations

import argparse
import sys
from pathlib import Path

import paramiko


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing env file: {path}")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description="Connect to an SSH server using credentials from .env.")
    parser.add_argument("--env-file", default=".env", help="Path to env file")
    parser.add_argument(
        "--command",
        default="hostname",
        help="Remote command to run after login",
    )
    args = parser.parse_args()

    env = load_env_file(Path(args.env_file))
    host = env.get("SSH_HOST", "")
    port = int(env.get("SSH_PORT", "22"))
    user = env.get("SSH_USER", "")
    password = env.get("SSH_PASSWORD", "")

    if not host or not user or not password:
        print("Missing SSH_HOST, SSH_USER, or SSH_PASSWORD in env file.", file=sys.stderr)
        return 2

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(
            hostname=host,
            port=port,
            username=user,
            password=password,
            timeout=10,
            auth_timeout=10,
            banner_timeout=10,
        )
        stdin, stdout, stderr = client.exec_command(args.command, timeout=10)
        output = stdout.read().decode("utf-8", errors="replace").strip()
        error = stderr.read().decode("utf-8", errors="replace").strip()
        print(f"Connected to {user}@{host}:{port}")
        if output:
            print(output)
        if error:
            print(error, file=sys.stderr)
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
