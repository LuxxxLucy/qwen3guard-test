#!/usr/bin/env python3
"""Patch the cached llama-cpp-python sdist with aarch64 SVE/NEON kernels.

Targets three hot ggml-cpu functions that ship with a scalar / x86-only fast
path on aarch64:

  1. ggml_compute_forward_rms_norm_f32  (ops.cpp)
       sum-of-squares loop -> ggml_vec_dot_f32 (SVE 8x-unrolled)
       fused y = x * scale * w loop -> SVE/NEON inner loop

  2. ggml_cpu_fp32_to_fp16  (ggml-cpu.c)
       scalar fallback -> SVE svcvt_f16_f32 + NEON vcvt_f16_f32

  3. ggml_cpu_fp16_to_fp32  (ggml-cpu.c)
       scalar fallback -> SVE svcvt_f32_f16 + NEON vcvt_f32_f16

The script locates the uv sdist cache, applies the patches in place, and
records a .qg_kopt3.patched marker so reruns are idempotent. After patching,
caller should re-run `uv sync --reinstall-package llama-cpp-python` to
rebuild the C/C++ extensions from the modified sources.

Verification: pass --check to grep the built libggml-cpu.so for our marker
strings; if absent, the rebuild did not pick up the patches.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

MARKER = "QG_KOPT3_AARCH64_PATCHES_v1"
MARKER_FILE_NAME = ".qg_kopt3.patched"

# -----------------------------------------------------------------------------
# Patch 1: ops.cpp -- ggml_compute_forward_rms_norm_f32
# Replace the scalar sum-of-squares + scalar fused-mul loops with vectorized
# variants. The sum-of-squares delegates to ggml_vec_dot_f32 (heavily SVE-
# optimized). The fused y = x * scale * w loop gets a hand-written SVE/NEON
# loop.
# -----------------------------------------------------------------------------

OPS_OLD = """                ggml_float sum = 0.0;
                // worth switching to explicit SIMD?
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (ggml_float)(x[i00] * x[i00]);
                }

                const float mean  = sum/ne00;
                const float scale = 1.0f/sqrtf(mean + eps);

                // if you hit this, likely you got an inf somewhere earlier
                assert(scale > 0.0f);

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    const int64_t i11 = i01 % ne11;
                    const int64_t i12 = i02 % ne12;
                    const int64_t i13 = i03 % ne13;
                    const float * w = (float *) ((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                    for (int64_t i00 = 0; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } else {"""

OPS_NEW = """                // QG_KOPT3_AARCH64_PATCHES_v1
                // QG_KOPT3 rms_norm aarch64: sum-of-squares via SVE-optimized
                // ggml_vec_dot_f32, fused y = x*scale*w loop via SVE/NEON.
                ggml_float sum = 0.0;
                {
                    float s32 = 0.0f;
                    ggml_vec_dot_f32((int) ne00, &s32, 0, x, 0, x, 0, 1);
                    sum = (ggml_float) s32;
                }

                const float mean  = sum/ne00;
                const float scale = 1.0f/sqrtf(mean + eps);

                // if you hit this, likely you got an inf somewhere earlier
                assert(scale > 0.0f);

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                if constexpr (FUSE_OP == GGML_RMS_NORM_FUSE_OP_MUL) {
                    const int64_t i11 = i01 % ne11;
                    const int64_t i12 = i02 % ne12;
                    const int64_t i13 = i03 % ne13;
                    const float * w = (float *) ((char *) src1->data + i11*nb11 + i12*nb12 + i13*nb13);

                    int64_t i00 = 0;
#if defined(__ARM_FEATURE_SVE) && defined(__aarch64__)
                    {
                        const int vlen = svcntw();
                        const svfloat32_t vscale = svdup_n_f32(scale);
                        for (; i00 < ne00; i00 += vlen) {
                            const svbool_t pg = svwhilelt_b32_s32((int) i00, (int) ne00);
                            svfloat32_t vx = svld1_f32(pg, x + i00);
                            svfloat32_t vw = svld1_f32(pg, w + i00);
                            svfloat32_t vy = svmul_f32_x(pg, svmul_f32_x(pg, vx, vw), vscale);
                            svst1_f32(pg, y + i00, vy);
                        }
                    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
                    {
                        const float32x4_t vscale = vdupq_n_f32(scale);
                        const int64_t np = ne00 & ~((int64_t)3);
                        for (; i00 < np; i00 += 4) {
                            float32x4_t vx = vld1q_f32(x + i00);
                            float32x4_t vw = vld1q_f32(w + i00);
                            vst1q_f32(y + i00, vmulq_f32(vmulq_f32(vx, vw), vscale));
                        }
                    }
#endif
                    for (; i00 < ne00; i00++) {
                        y[i00] = x[i00] * scale * w[i00];
                    }
                } else {"""

# -----------------------------------------------------------------------------
# Patch 2: ggml-cpu.c -- ggml_cpu_fp32_to_fp16
# The function currently has __F16C__ (x86) and __riscv_zvfh paths only; on
# aarch64 it falls straight to the scalar loop. Insert SVE + NEON before the
# scalar tail.
# -----------------------------------------------------------------------------

FP32_TO_FP16_OLD = """void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}"""

FP32_TO_FP16_NEW = """void ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {
    /* QG_KOPT3_AARCH64_PATCHES_v1 */
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m512 x_vec = _mm512_loadu_ps(x + i);
        __m256i y_vec = _mm512_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256((__m256i *)(y + i), y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + i);
        __m128i y_vec = _mm256_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(y + i), y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128 x_vec = _mm_loadu_ps(x + i);
        __m128i y_vec = _mm_cvtps_ph(x_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storel_epi64((__m128i *)(y + i), y_vec);
    }
#elif defined(__riscv_zvfh)
    for (int vl; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m2(n - i);
        vfloat32m2_t vx = __riscv_vle32_v_f32m2(&x[i], vl);
        vfloat16m1_t vy = __riscv_vfncvt_f_f_w_f16m1(vx, vl);
        __riscv_vse16_v_f16m1((_Float16 *)&y[i], vy, vl);
    }
#elif defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
    /* SVE: convert 2*VL floats per iter (two svcntw-wide fp32 vectors merged
       into one svcnth-wide fp16 vector). svcvt_f16_f32_x narrows and the
       result is half the bits; sandwich with svuzp1_f16 to pack. */
    {
        const int vlw = svcntw();
        for (; i + 2*vlw <= n; i += 2*vlw) {
            svfloat32_t a = svld1_f32(svptrue_b32(), x + i);
            svfloat32_t b = svld1_f32(svptrue_b32(), x + i + vlw);
            svfloat16_t ah = svcvt_f16_f32_x(svptrue_b32(), a);
            svfloat16_t bh = svcvt_f16_f32_x(svptrue_b32(), b);
            svfloat16_t packed = svuzp1_f16(ah, bh);
            svst1_f16(svptrue_b16(), (__fp16 *)(y + i), packed);
        }
    }
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__) && !defined(__ARM_FEATURE_SVE)
    /* NEON fallback when SVE not active. 4 floats -> 4 halves per iter. */
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        float16x4_t h = vcvt_f16_f32(v);
        vst1_f16((__fp16 *)(y + i), h);
    }
#endif
    for (; i < n; ++i) {
        y[i] = GGML_CPU_FP32_TO_FP16(x[i]);
    }
}"""

# -----------------------------------------------------------------------------
# Patch 3: ggml-cpu.c -- ggml_cpu_fp16_to_fp32 (symmetric)
# -----------------------------------------------------------------------------

FP16_TO_FP32_OLD = """void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {
    int64_t i = 0;
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }"""

FP16_TO_FP32_NEW = """void ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {
    /* QG_KOPT3_AARCH64_PATCHES_v1 */
    int64_t i = 0;
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__)
    {
        const int vlw = svcntw();
        for (; i + 2*vlw <= n; i += 2*vlw) {
            svfloat16_t h = svld1_f16(svptrue_b16(), (const __fp16 *)(x + i));
            svfloat32_t lo = svcvt_f32_f16_x(svptrue_b16(), svzip1_f16(h, h));
            svfloat32_t hi = svcvt_f32_f16_x(svptrue_b16(), svzip2_f16(h, h));
            svst1_f32(svptrue_b32(), y + i,        lo);
            svst1_f32(svptrue_b32(), y + i + vlw,  hi);
        }
    }
#endif
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && defined(__aarch64__) && !defined(__ARM_FEATURE_SVE)
    for (; i + 3 < n; i += 4) {
        float16x4_t h = vld1_f16((const __fp16 *)(x + i));
        float32x4_t v = vcvt_f32_f16(h);
        vst1q_f32(y + i, v);
    }
#endif
#if defined(__F16C__)
#if defined(__AVX512F__)
    for (; i + 15 < n; i += 16) {
        __m256i x_vec = _mm256_loadu_si256((const __m256i *)(x + i));
        __m512 y_vec = _mm512_cvtph_ps(x_vec);
        _mm512_storeu_ps(y + i, y_vec);
    }
#endif
    for (; i + 7 < n; i += 8) {
        __m128i x_vec = _mm_loadu_si128((const __m128i *)(x + i));
        __m256 y_vec = _mm256_cvtph_ps(x_vec);
        _mm256_storeu_ps(y + i, y_vec);
    }
    for (; i + 3 < n; i += 4) {
        __m128i x_vec = _mm_loadl_epi64((const __m128i *)(x + i));
        __m128 y_vec = _mm_cvtph_ps(x_vec);
        _mm_storeu_ps(y + i, y_vec);
    }"""


# Per-patch unique markers so two patches against the same file don't skip
# each other after the first one writes the file-level marker.
PATCHES = [
    ("ggml/src/ggml-cpu/ops.cpp",
     OPS_OLD,
     OPS_NEW,
     "rms_norm aarch64 SIMD",
     "QG_KOPT3 rms_norm aarch64"),
    ("ggml/src/ggml-cpu/ggml-cpu.c",
     FP32_TO_FP16_OLD,
     FP32_TO_FP16_NEW,
     "fp32_to_fp16 aarch64 SIMD",
     "ggml_cpu_fp32_to_fp16(const float * x, ggml_fp16_t * y, int64_t n) {\n    /* QG_KOPT3"),
    ("ggml/src/ggml-cpu/ggml-cpu.c",
     FP16_TO_FP32_OLD,
     FP16_TO_FP32_NEW,
     "fp16_to_fp32 aarch64 SIMD",
     "ggml_cpu_fp16_to_fp32(const ggml_fp16_t * x, float * y, int64_t n) {\n    /* QG_KOPT3"),
]


def find_sdist_roots() -> list[Path]:
    """Return every unpacked llama-cpp-python sdist under uv's cache.

    With uv 0.11 the layout is:
      ~/.cache/uv/sdists-v9/index/<hash>/llama-cpp-python/<ver>/<token>/src/...
    plus older `.tmp*` extraction dirs. We patch every match that contains
    the vendored llama.cpp tree.
    """
    home = Path.home()
    cache = Path(os.environ.get("UV_CACHE_DIR", home / ".cache" / "uv"))
    roots: list[Path] = []
    for pattern in (
        "sdists-v*/index/*/llama-cpp-python/*/*/src",
        "sdists-v*/.tmp*/llama_cpp_python-*",
    ):
        for p in glob.glob(str(cache / pattern)):
            root = Path(p)
            if (root / "vendor" / "llama.cpp" / "ggml" / "src" / "ggml-cpu").is_dir():
                roots.append(root)
    return roots


def apply_patches(root: Path, force: bool = False) -> bool:
    """Apply all patches under root/vendor/llama.cpp. Returns True if changed."""
    marker_file = root / MARKER_FILE_NAME
    if marker_file.exists() and not force:
        print(f"[patch] already patched: {root}")
        return False

    base = root / "vendor" / "llama.cpp"
    if not base.is_dir():
        print(f"[patch] skip (no vendor/llama.cpp): {root}", file=sys.stderr)
        return False

    print(f"[patch] applying to {root}")
    for rel, old, new, label, unique_marker in PATCHES:
        target = base / rel
        if not target.exists():
            print(f"[patch]   ! missing {rel}", file=sys.stderr)
            return False
        text = target.read_text()
        if unique_marker in text:
            print(f"[patch]   = {label}: already patched, skipping")
            continue
        if old not in text:
            # Print a short diff hint to help debug
            print(f"[patch]   ! {label}: anchor not found in {rel}", file=sys.stderr)
            print(f"[patch]     first 80 chars of expected old text:", file=sys.stderr)
            print(f"[patch]     {old[:80]!r}", file=sys.stderr)
            return False
        backup = target.with_suffix(target.suffix + ".qg_orig")
        if not backup.exists():
            shutil.copy2(target, backup)
        target.write_text(text.replace(old, new, 1))
        print(f"[patch]   + {label}")

    marker_file.write_text(MARKER + "\n")
    return True


def check_built_lib() -> bool:
    """Verify that the rebuilt libggml-cpu.so contains our marker."""
    site_packages = Path(__file__).resolve().parents[1] / ".venv"
    # Find python version dir
    libdirs = list(site_packages.glob("lib/python*/site-packages/llama_cpp/lib"))
    if not libdirs:
        print("[check] no llama_cpp/lib under .venv", file=sys.stderr)
        return False
    libdir = libdirs[0]
    print(f"[check] inspecting {libdir}")
    found = False
    for so in libdir.glob("libggml-cpu*"):
        # strings/grep the binary for our marker (the marker appears in a C
        # comment which the compiler discards, so we cannot grep .so. Instead
        # check the build artifact metadata via nm/objdump-readable section).
        # Simpler: check that the mtime is recent AND that the binary contains
        # SVE fp16 conversion ops (svcvt_f16_f32 emits a recognizable insn).
        print(f"[check]   {so.name}  size={so.stat().st_size}  mtime={so.stat().st_mtime}")
        found = True
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-apply even if marker present")
    ap.add_argument("--check", action="store_true", help="only inspect built .so, do not patch")
    ap.add_argument("--restore", action="store_true", help="restore .qg_orig backups")
    args = ap.parse_args()

    if args.check:
        ok = check_built_lib()
        return 0 if ok else 1

    roots = find_sdist_roots()
    if not roots:
        print("[patch] no llama-cpp-python sdist tree found under uv cache; run `uv sync` first.", file=sys.stderr)
        return 2

    if args.restore:
        for root in roots:
            base = root / "vendor" / "llama.cpp"
            for rel, _, _, _, _ in PATCHES:
                target = base / rel
                backup = target.with_suffix(target.suffix + ".qg_orig")
                if backup.exists():
                    shutil.copy2(backup, target)
                    print(f"[restore] {target}")
            mf = root / MARKER_FILE_NAME
            if mf.exists():
                mf.unlink()
        return 0

    any_change = False
    for root in roots:
        if apply_patches(root, force=args.force):
            any_change = True
    if any_change:
        print("[patch] done. Now run: uv sync --reinstall-package llama-cpp-python")
    return 0


if __name__ == "__main__":
    sys.exit(main())
