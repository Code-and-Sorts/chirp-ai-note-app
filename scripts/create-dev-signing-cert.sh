#!/usr/bin/env bash
set -euo pipefail

CERT_NAME="${1:-Chirp Dev}"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"

if security find-identity -v -p codesigning | grep -qF "$CERT_NAME"; then
    echo "Code-signing identity '$CERT_NAME' already exists."
    echo "Build with it: export CHIRP_CODESIGN_IDENTITY=\"$CERT_NAME\""
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

openssl req -x509 -newkey rsa:2048 -sha256 -days 3650 -nodes \
    -keyout "$TMP/key.pem" -out "$TMP/cert.pem" \
    -subj "/CN=$CERT_NAME" \
    -addext "basicConstraints=critical,CA:FALSE" \
    -addext "keyUsage=critical,digitalSignature" \
    -addext "extendedKeyUsage=critical,codeSigning"

openssl pkcs12 -export -out "$TMP/cert.p12" \
    -inkey "$TMP/key.pem" -in "$TMP/cert.pem" -passout pass:

security import "$TMP/cert.p12" -k "$KEYCHAIN" -P "" -T /usr/bin/codesign

if ! security find-identity -v -p codesigning | grep -qF "$CERT_NAME"; then
    echo "error: '$CERT_NAME' was imported but is not a valid code-signing identity." >&2
    echo "Open Keychain Access, find '$CERT_NAME', and set its trust to 'Always Trust'." >&2
    exit 1
fi

echo "Created code-signing identity '$CERT_NAME'."
echo
echo "1. Add to your shell profile:  export CHIRP_CODESIGN_IDENTITY=\"$CERT_NAME\""
echo "2. Rebuild:                    uv run python -m audio_capture.build"
echo "3. Grant Microphone + Screen Recording one last time."
echo "   Future rebuilds keep the grant."
