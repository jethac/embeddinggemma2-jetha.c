#!/bin/sh

set -eu

repo="jethac/embeddinggemma2-jetha.c"
version=${EMBEDDINGGEMMA2_VERSION:-latest}
variant=${EMBEDDINGGEMMA2_VARIANT:-auto}
install_dir=${EMBEDDINGGEMMA2_INSTALL_DIR:-}

usage() {
    cat <<'EOF'
Usage: install.sh [--version VERSION] [--variant auto|cpu|metal|cuda|rocm|xpu]
                  [--install-dir DIRECTORY]

Downloads a release binary and its matching runtime archive and installs it as
~/.local/bin/embeddinggemma2-jetha by default.

Environment overrides:
  EMBEDDINGGEMMA2_VERSION       Release tag, or latest (default: latest)
  EMBEDDINGGEMMA2_VARIANT       auto, cpu, metal, cuda, rocm, or xpu (default: auto)
  EMBEDDINGGEMMA2_INSTALL_DIR   Installation directory (default: ~/.local/bin)
EOF
}

die() {
    printf 'embeddinggemma2-jetha installer: %s\n' "$*" >&2
    exit 1
}

need_value() {
    [ "$#" -ge 2 ] || die "$1 requires a value"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --version)
            need_value "$@"
            version=$2
            shift 2
            ;;
        --variant)
            need_value "$@"
            variant=$2
            shift 2
            ;;
        --install-dir)
            need_value "$@"
            install_dir=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

case "$version" in
    ''|*[!A-Za-z0-9._-]*) die "invalid release version: $version" ;;
esac

case "$variant" in
    auto|cpu|metal|cuda|rocm|xpu) ;;
    *) die "invalid variant: $variant" ;;
esac

if [ -z "$install_dir" ]; then
    [ -n "${HOME:-}" ] || die 'HOME is not set; use --install-dir'
    install_dir=$HOME/.local/bin
fi

case "$(uname -s)" in
    Darwin) platform=darwin ;;
    Linux) platform=linux ;;
    *) die "unsupported operating system: $(uname -s)" ;;
esac

case "$(uname -m)" in
    arm64|aarch64) architecture=arm64 ;;
    x86_64|amd64) architecture=x86_64 ;;
    *) die "unsupported architecture: $(uname -m)" ;;
esac

case "$platform-$architecture" in
    darwin-arm64|linux-x86_64|linux-arm64) ;;
    *)
        die "no published binary for $platform-$architecture; build from source"
        ;;
esac

requested_variant=$variant
if [ "$variant" = auto ]; then
    if [ "$platform" = darwin ]; then
        variant=metal
    elif command -v nvidia-smi >/dev/null 2>&1 &&
         nvidia-smi -L >/dev/null 2>&1; then
        variant=cuda
    elif [ -e /dev/kfd ]; then
        variant=rocm
    elif command -v sycl-ls >/dev/null 2>&1 &&
         sycl-ls 2>/dev/null | grep -Eq '\[level_zero:gpu\]'; then
        variant=xpu
    else
        variant=cpu
    fi
fi

# ARM64 CUDA covers the GB10 target; ROCm/XPU ARM assets are not part of the matrix.
if [ "$platform-$architecture" = linux-arm64 ] && [ "$variant" != cpu ] && [ "$variant" != cuda ]; then
    if [ "$requested_variant" = auto ]; then
        variant=cpu
    else
        die "the $variant variant is not published for linux-arm64; use cpu or cuda"
    fi
fi

case "$platform-$variant" in
    darwin-cpu|darwin-metal|linux-cpu|linux-cuda|linux-rocm|linux-xpu) ;;
    *) die "the $variant variant is not available for $platform-$architecture" ;;
esac

if [ "$version" = latest ]; then
    release_url="https://github.com/$repo/releases/latest/download"
else
    release_url="https://github.com/$repo/releases/download/$version"
fi

download() {
    url=$1
    output=$2
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 --retry-delay 1 --connect-timeout 15 \
            -o "$output" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -q --tries=3 --timeout=15 -O "$output" "$url"
    else
        die 'curl or wget is required'
    fi
}

sha256_file() {
    file=$1
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$file" | awk '{print $NF}'
    else
        die 'sha256sum, shasum, or openssl is required'
    fi
}

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/embeddinggemma2-jetha.XXXXXX") ||
    die 'could not create a temporary directory'
staged=
cleanup() {
    rm -rf "$tmpdir"
    if [ -n "$staged" ]; then
        rm -f "$staged"
    fi
}
trap cleanup EXIT HUP INT TERM

checksums=$tmpdir/SHA256SUMS
download "$release_url/SHA256SUMS" "$checksums" ||
    die "could not download this project's release checksums; use the README's CMake source installation"

verify_asset() {
    verify_name=$1
    verify_file=$tmpdir/$verify_name
    download "$release_url/$verify_name" "$verify_file"
    expected=$(awk -v name="$verify_name" '$2 == name { print $1; count++ } END { if (count != 1) exit 1 }' "$checksums") ||
        die "SHA256SUMS must contain exactly one entry for $verify_name"
    case "$expected" in ''|*[!a-fA-F0-9]*) die "invalid checksum for $verify_name" ;; esac
    [ "${#expected}" -eq 64 ] || die "invalid checksum for $verify_name"
    actual=$(sha256_file "$verify_file")
    [ "$actual" = "$expected" ] || die "checksum verification failed for $verify_name"
}

prepare_variant() {
    asset="embeddinggemma2-jetha-$platform-$architecture-$variant"
    runtime="$asset.runtime.tar.gz"
    printf 'Selecting %s (%s/%s, %s)\n' "$asset" "$platform" "$architecture" "$variant"
    verify_asset "$asset"
    verify_asset "$runtime"
    # A checksum-verified runtime archive must stay inside its application prefix.
    tar -tzf "$tmpdir/$runtime" > "$tmpdir/entries"
    awk '
        /^\// || /(^|\/)\.\.(\/|$)/ { exit 1 }
        !/^(bin|lib|share)\// { exit 1 }
    ' "$tmpdir/entries" || die 'runtime archive contains unsafe paths'
    tar -tvzf "$tmpdir/$runtime" > "$tmpdir/entry-types"
    awk '
        $1 !~ /^[-dl]/ { exit 1 }
        $1 ~ /^l/ && ($(NF-1) != "->" || $NF !~ /^[A-Za-z0-9._+-]+$/ || $NF == "." || $NF == "..") { exit 1 }
    ' "$tmpdir/entry-types" || die 'runtime archive contains unsafe links or special files'
    app=$tmpdir/$asset.app
    mkdir "$app"
    tar -xzf "$tmpdir/$runtime" -C "$app"
    mkdir -p "$app/bin"
    cp "$tmpdir/$asset" "$app/bin/embeddinggemma2-jetha"
    chmod 0755 "$app/bin/embeddinggemma2-jetha"
    [ -f "$app/share/licenses/embeddinggemma2-jetha/LICENSE" ] || die 'runtime archive has no project license'
}
prepare_variant

# A visible accelerator does not guarantee that the published binary's runtime
# libraries are installed. In automatic mode, use CPU if the loader reports a
# missing dependency.
if [ "$platform" = linux ] && command -v ldd >/dev/null 2>&1; then
    ldd_output=$(ldd "$app/bin/embeddinggemma2-jetha" 2>&1 || true)
    if printf '%s\n' "$ldd_output" | grep -q 'not found'; then
        if [ "$requested_variant" = auto ] && [ "$variant" != cpu ]; then
            printf '%s\n' \
                "$variant runtime dependencies are missing; falling back to the CPU binary." >&2
            variant=cpu
            prepare_variant
            ldd_output=$(ldd "$app/bin/embeddinggemma2-jetha" 2>&1 || true)
        fi
        if printf '%s\n' "$ldd_output" | grep -q 'not found'; then
            printf '%s\n' "$ldd_output" >&2
            die "$asset has unavailable shared-library dependencies"
        fi
    fi
fi

"$app/bin/embeddinggemma2-jetha" --help >/dev/null 2>&1 || die 'staged executable failed to load'
mkdir -p "$install_dir"
application=$(mktemp -d "$install_dir/.embeddinggemma2-jetha-runtime.XXXXXX") ||
    die "could not create an application directory in $install_dir"
cp -R "$app/." "$application/"
staged=$(mktemp "$install_dir/.embeddinggemma2-jetha.XXXXXX") ||
    die "could not create a temporary launcher in $install_dir"
rm -f "$staged"
ln -s "$(basename "$application")/bin/embeddinggemma2-jetha" "$staged"
mv -f "$staged" "$install_dir/embeddinggemma2-jetha"
staged=

printf 'Installed %s as %s/embeddinggemma2-jetha\n' "$asset" "$install_dir"
case ":${PATH:-}:" in
    *":$install_dir:"*) ;;
    *) printf 'Add %s to PATH to run embeddinggemma2-jetha directly.\n' "$install_dir" ;;
esac
