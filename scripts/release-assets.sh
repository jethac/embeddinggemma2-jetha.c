#!/bin/sh

set -eu

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: scripts/release-assets.sh checksums [DIST_DIR]
       scripts/release-assets.sh verify [DIST_DIR]

Writes or verifies the complete port release: raw executables and runtime archives.
EOF
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        openssl dgst -sha256 "$1" | awk '{print $NF}'
    else
        die 'sha256sum, shasum, or openssl is required'
    fi
}

[ "$#" -ge 1 ] && [ "$#" -le 2 ] || {
    usage >&2
    exit 1
}

command=$1
dist_dir=${2:-dist}
targets='darwin-arm64-cpu
darwin-arm64-metal
linux-arm64-cpu
linux-arm64-cuda
linux-x86_64-cpu
linux-x86_64-cuda
linux-x86_64-rocm
linux-x86_64-xpu
windows-x86_64-cpu'
assets=$(printf '%s\n' "$targets" | while IFS= read -r target; do
    case "$target" in
        windows-*) printf 'embeddinggemma2-jetha-%s.exe\nembeddinggemma2-jetha-%s.runtime.zip\n' "$target" "$target" ;;
        *) printf 'embeddinggemma2-jetha-%s\nembeddinggemma2-jetha-%s.runtime.tar.gz\n' "$target" "$target" ;;
    esac
done)
asset_count=$(printf '%s\n' "$assets" | wc -l | tr -d ' ')

require_assets() {
    printf '%s\n' "$assets" | while IFS= read -r asset; do
        path=$dist_dir/$asset
        [ -f "$path" ] || die "missing release asset: $path"
        [ ! -L "$path" ] || die "release asset must not be a symlink: $path"
        [ -s "$path" ] || die "release asset is empty: $path"
        case "$asset" in
            *.runtime.tar.gz) tar -tzf "$path" >/dev/null || die "invalid runtime archive: $asset"; continue ;;
            *.runtime.zip) unzip -tq "$path" >/dev/null || die "invalid runtime archive: $asset"; continue ;;
            *.exe) ;;
            *) [ -x "$path" ] || die "release asset is not executable: $path" ;;
        esac

        description=$(file -b "$path")
        case "$asset" in
            embeddinggemma2-jetha-darwin-arm64-*)
                printf '%s\n' "$description" | grep -Eq 'Mach-O.*arm64' ||
                    die "$asset is not an arm64 Mach-O executable: $description"
                ;;
            embeddinggemma2-jetha-linux-x86_64-*)
                printf '%s\n' "$description" | grep -Eq 'ELF 64-bit.*x86-64' ||
                    die "$asset is not an x86-64 ELF executable: $description"
                ;;
            embeddinggemma2-jetha-linux-arm64-*)
                printf '%s\n' "$description" | grep -Eq 'ELF 64-bit.*(aarch64|ARM aarch64)' ||
                    die "$asset is not an arm64 ELF executable: $description"
                ;;
            embeddinggemma2-jetha-windows-x86_64-*)
                printf '%s\n' "$description" | grep -Eq 'PE32\+.*x86-64' ||
                    die "$asset is not an x86-64 PE executable: $description"
                ;;
        esac
    done
}

case "$command" in
    checksums)
        require_assets
        temporary=$dist_dir/.SHA256SUMS.tmp
        trap 'rm -f "$temporary"' EXIT HUP INT TERM
        : > "$temporary"
        printf '%s\n' "$assets" | while IFS= read -r asset; do
            printf '%s  %s\n' "$(sha256_file "$dist_dir/$asset")" "$asset"
        done >> "$temporary"
        mv -f "$temporary" "$dist_dir/SHA256SUMS"
        trap - EXIT HUP INT TERM
        printf 'Wrote %s/SHA256SUMS\n' "$dist_dir"
        ;;
    verify)
        require_assets
        checksums=$dist_dir/SHA256SUMS
        [ -f "$checksums" ] || die "missing release asset: $checksums"
        [ "$(awk 'NF { count++ } END { print count + 0 }' "$checksums")" -eq "$asset_count" ] ||
            die "SHA256SUMS must contain exactly $asset_count entries"
        printf '%s\n' "$assets" | while IFS= read -r asset; do
            expected=$(awk -v name="$asset" '$2 == name { print $1; count++ } END { if (count != 1) exit 1 }' "$checksums") ||
                die "SHA256SUMS must contain exactly one entry for $asset"
            actual=$(sha256_file "$dist_dir/$asset")
            [ "$actual" = "$expected" ] || die "checksum mismatch: $asset"
        done
        printf 'Verified %s release assets and %s/SHA256SUMS\n' "$asset_count" "$dist_dir"
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
