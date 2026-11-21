#!/bin/sh
# Install servecheck from GitHub Releases with SHA-256 verification.
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/SiluPanda/servecheck/main/install.sh | sh
#   VERSION=v0.1.0 INSTALL_DIR=/usr/local/bin sh install.sh
set -eu

REPO="SiluPanda/servecheck"
BIN="servecheck"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
VERSION="${VERSION:-latest}"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command '$1' not found" >&2
    exit 1
  fi
}

need_cmd curl
need_cmd tar

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
case "$OS" in
  darwin|linux) ;;
  *)
    echo "error: unsupported OS '$OS' (supported: darwin, linux)" >&2
    exit 1
    ;;
esac

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ARCH="amd64" ;;
  arm64|aarch64) ARCH="arm64" ;;
  *)
    echo "error: unsupported architecture '$ARCH' (supported: amd64, arm64)" >&2
    exit 1
    ;;
esac

if [ "$VERSION" = "latest" ]; then
  TAG="$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
    | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)"
  if [ -z "$TAG" ]; then
    echo "error: failed to determine latest release tag for ${REPO}" >&2
    exit 1
  fi
else
  case "$VERSION" in
    v*) TAG="$VERSION" ;;
    *) TAG="v$VERSION" ;;
  esac
fi

ARCHIVE="${BIN}_${TAG}_${OS}_${ARCH}.tar.gz"
BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"

TMPDIR_WORK="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_WORK"' EXIT

echo "Downloading ${ARCHIVE} (${TAG})..."
curl -fsSL "${BASE_URL}/${ARCHIVE}" -o "${TMPDIR_WORK}/${ARCHIVE}"
curl -fsSL "${BASE_URL}/checksums.txt" -o "${TMPDIR_WORK}/checksums.txt"

EXPECTED="$(awk -v f="$ARCHIVE" '$2 == f || $2 == ("*" f) { print $1; exit }' "${TMPDIR_WORK}/checksums.txt")"
if [ -z "$EXPECTED" ]; then
  echo "error: checksum entry for ${ARCHIVE} not found in checksums.txt" >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL="$(sha256sum "${TMPDIR_WORK}/${ARCHIVE}" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  ACTUAL="$(shasum -a 256 "${TMPDIR_WORK}/${ARCHIVE}" | awk '{print $1}')"
else
  echo "error: neither sha256sum nor shasum found; refusing unverified install" >&2
  exit 1
fi

if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "error: SHA-256 mismatch for ${ARCHIVE}" >&2
  echo "  expected: ${EXPECTED}" >&2
  echo "  actual:   ${ACTUAL}" >&2
  exit 1
fi

mkdir -p "$INSTALL_DIR"
tar -xzf "${TMPDIR_WORK}/${ARCHIVE}" -C "${TMPDIR_WORK}"
install -m 0755 "${TMPDIR_WORK}/${BIN}" "${INSTALL_DIR}/${BIN}"

echo "Installed ${BIN} (${TAG}) to ${INSTALL_DIR}/${BIN}"
case ":$PATH:" in
  *":${INSTALL_DIR}:"*) ;;
  *) echo "Note: add '${INSTALL_DIR}' to your PATH to run '${BIN}' directly." ;;
esac
"${INSTALL_DIR}/${BIN}" version
