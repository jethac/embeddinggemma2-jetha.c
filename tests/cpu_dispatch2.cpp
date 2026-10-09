// Safety invariant: CPU instruction support alone cannot authorize a SIMD backend.
// Exercise the pinned selector with synthetic CPUID/XCR0 states; no AVX instructions
// are compiled into this check, so it can run on baseline x86-64 CI machines.
#include "ggml-cpu/arch/x86/cpu-feats.cpp"
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--expect-disabled") == 0 &&
        ggml_backend_cpu_x86_score() != 0) {
        std::fprintf(stderr, "SIMD backend was enabled on an unsupported emulated CPU\n");
        return 1;
    }
    cpuid_x86 cpu;
    cpu.f_1_ecx.set();
    cpu.f_7_ebx.set();
    cpu.f_7_ecx.set();
    cpu.f_7_1_eax.set();
    auto check = [&](uint64_t state, bool usable, const char * name) {
        bool actual = ggml_backend_cpu_x86_score_features(cpu, state) > 0;
        if (actual != usable) {
            std::fprintf(stderr, "%s: unexpected CPU selector result\n", name);
            return false;
        }
        return true;
    };
    bool ok = check(0xe6, true, "complete OS state");
#ifdef GGML_AVX
    ok &= check(0, false, "no OS vector state");
    ok &= check(2, false, "missing YMM state");
    cpu.f_1_ecx.reset(27);
    ok &= check(0xe6, false, "OSXSAVE disabled");
    cpu.f_1_ecx.set(27);
    cpu.f_1_ecx.reset(26);
    ok &= check(0xe6, false, "XSAVE unavailable");
    cpu.f_1_ecx.set(26);
    cpu.f_1_ecx.reset(28);
    ok &= check(0xe6, false, "AVX unavailable");
    cpu.f_1_ecx.set(28);
#else
    cpu.f_1_ecx.reset();
    cpu.f_7_ebx.reset();
    ok &= check(0, true, "baseline fallback");
#endif
#ifdef GGML_AVX512
    ok &= check(6, false, "AVX without ZMM state");
    ok &= check(0x66, false, "missing high ZMM state");
    ok &= check(0xc6, false, "missing opmask state");
    cpu.f_7_ebx.reset(16);
    ok &= check(0xe6, false, "AVX512F unavailable");
#elif defined(GGML_AVX)
    ok &= check(6, true, "AVX OS state");
#endif
    if (ok) std::puts("CPU/OS dispatch checks passed");
    return ok ? 0 : 1;
}
