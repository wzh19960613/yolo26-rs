#!/bin/sh
set -eu

target="web"
dir=""
name="yolo26"
features="wasm classify segment semantic pose obb yoloe-visual yoloe-pf default_labels"

usage() {
  cat <<'EOF'
Usage: build-wasm.sh [OPTIONS]

Options:
  --target TARGET   wasm-pack target (default: web)
  --dir DIR         copy output files to DIR after build
  --name NAME       output name (default: yolo26)
  --features LIST   cargo features passed to wasm-pack
                    (default: wasm classify segment semantic pose obb
                     yoloe-visual yoloe-pf default_labels)
  --help            show this help message
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --target) target="$2"; shift 2 ;;
    --dir)    dir="$2";    shift 2 ;;
    --name)   name="$2";   shift 2 ;;
    --features) features="$2"; shift 2 ;;
    --help)   usage; exit 0 ;;
    *)        echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

script_dir="$(cd "$(dirname "$0")" && pwd)"
outdir="${script_dir}/target/wasm-pack-${target}"

wasm-pack build "$script_dir" \
  --target "$target" \
  --out-dir "$outdir" \
  --out-name "$name" \
  --features "$features"

if [ -n "$dir" ]; then
  mkdir -p "$dir"
  for f in "$outdir"/${name}*; do
    [ -f "$f" ] && cp "$f" "$dir/"
  done
fi
