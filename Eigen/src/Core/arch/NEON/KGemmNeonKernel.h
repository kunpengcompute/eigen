#ifndef KGEMM_NEON_KERNEL_H
#define KGEMM_NEON_KERNEL_H

#if defined(__clang__)
#define vldr(vd, addr, offset) \
    asm volatile("ldr q" #vd ", [%0, #" #offset "]" : : "r"(addr) : "memory", "v" #vd)

#define vldr_post(vd, addr) \
    asm volatile("ldr q" #vd ", [%0], #16" : "+r"(addr) : "r"(addr) : "memory", "v" #vd)

#define vldr_f64(vd, addr) \
    asm volatile("ldr d" #vd ", [%0]" : : "r"(addr) : "memory", "v" #vd)

#define vldr_f64_post(vd, addr) \
    asm volatile("ldr d" #vd ", [%0], #8" : "+r"(addr) : "r"(addr) : "memory", "v" #vd)

#define vldr_f32_post(vd, addr) \
    asm volatile("ldr s" #vd ", [%0], #4" : "+r"(addr) : "r"(addr) : "memory", "v" #vd)

#define vstr(vd, addr, offset) \
    asm volatile("str q" #vd ", [%0, #" #offset "]" : : "r"(addr) : "memory", "v" #vd)

#define vldp(vd0, vd1, addr, offset) \
    asm volatile("ldp q" #vd0 ", q" #vd1 ", [%0, #" #offset "]" : : "r"(addr) : "memory", "v" #vd0, "v" #vd1)

#define vldp_post(vd0, vd1, addr) \
    asm volatile("ldp q" #vd0 ", q" #vd1 ", [%0], #32" : "+r"(addr) : : "memory", "v" #vd0, "v" #vd1)

#elif defined(__GNUC__)

#define vldr(vd, addr, offset) \
    asm volatile("ldr q" #vd ", [\%x0, #" #offset "]" : : "r"(addr) : "memory", "v" #vd)

#define vldr_post(vd, addr) \
    asm volatile("ldr q" #vd ", [\%x0], #16" : "+r"(addr) : "r"(addr) : "memory", "v" #vd)

#define vldr_f64(vd, addr) \
    asm volatile("ldr d" #vd ", [\%x0]" : : "r"(addr) : "memory", "v" #vd)

#define vldr_f64_post(vd, addr) \
    asm volatile("ldr d" #vd ", [\%x0], #8" : "+r"(addr) : "r"(addr) : "memory", "v" #vd)

#define vldr_f32_post(vd, addr) \
    asm volatile("ldr s" #vd ", [\%x0], #4" : "+r"(addr) : "r"(addr) : "memory", "v" #vd)

#define vstr(vd, addr, offset) \
    asm volatile("str q" #vd ", [\%x0, #" #offset "]" : : "r"(addr) : "memory", "v" #vd)

#define vldp(vd0, vd1, addr, offset) \
    asm volatile("ldp q" #vd0 ", q" #vd1 ", [\%x0, #" #offset "]" : : "r"(addr) : "memory", "v" #vd0, "v" #vd1)

#define vldp_post(vd0, vd1, addr) \
    asm volatile("ldp q" #vd0 ", q" #vd1 ", [\%x0], #32" : "+r"(addr) : : "memory", "v" #vd0, "v" #vd1)

#endif

#define vldr_m4(vd0, vd1, vd2, vd3, addr, off0, off1, off2, off3) \
    vldr(vd0, addr, off0); \
    vldr(vd1, addr, off1); \
    vldr(vd2, addr, off2); \
    vldr(vd3, addr, off3);

#define vldr_post_m2(vd0, vd1, addr0, addr1) \
    vldr_post(vd0, addr0); \
    vldr_post(vd1, addr1);

#define vldr_post_m3(vd0, vd1, vd2, addr0, addr1, addr2) \
    vldr_post(vd0, addr0); \
    vldr_post(vd1, addr1); \
    vldr_post(vd2, addr2);

#define vldr_post_m4(vd0, vd1, vd2, vd3, addr0, addr1, addr2, addr3) \
    vldr_post(vd0, addr0); \
    vldr_post(vd1, addr1); \
    vldr_post(vd2, addr2); \
    vldr_post(vd3, addr3);

#define vldr_f64_post_m2(vd0, vd1, addr0, addr1) \
    vldr_f64_post(vd0, addr0); \
    vldr_f64_post(vd1, addr1);

#define vldr_f64_post_m3(vd0, vd1, vd2, addr0, addr1, addr2) \
    vldr_f64_post(vd0, addr0); \
    vldr_f64_post(vd1, addr1); \
    vldr_f64_post(vd2, addr2);

#define vldr_f64_post_m4(vd0, vd1, vd2, vd3, addr0, addr1, addr2, addr3) \
    vldr_f64_post(vd0, addr0); \
    vldr_f64_post(vd1, addr1); \
    vldr_f64_post(vd2, addr2); \
    vldr_f64_post(vd3, addr3);

#define vldr_f32_post_m2(vd0, vd1, addr0, addr1) \
    vldr_f32_post(vd0, addr0); \
    vldr_f32_post(vd1, addr1);

#define vldr_f32_post_m3(vd0, vd1, vd2, addr0, addr1, addr2) \
    vldr_f32_post(vd0, addr0); \
    vldr_f32_post(vd1, addr1); \
    vldr_f32_post(vd2, addr2);

#define vldr_f32_post_m4(vd0, vd1, vd2, vd3, addr0, addr1, addr2, addr3) \
    vldr_f32_post(vd0, addr0); \
    vldr_f32_post(vd1, addr1); \
    vldr_f32_post(vd2, addr2); \
    vldr_f32_post(vd3, addr3);

#define vldp_post_m2(vd0, vd1, vn0, vn1, addr0, addr1) \
    vldp_post(vd0, vn0, addr0); \
    vldp_post(vd1, vn1, addr1);

#define vldp_post_m3(vd0, vd1, vd2, vn0, vn1, vn2, addr0, addr1, addr2) \
    vldp_post(vd0, vn0, addr0); \
    vldp_post(vd1, vn1, addr1); \
    vldp_post(vd2, vn2, addr2);

#define vldp_post_m4(vd0, vd1, vd2, vd3, vn0, vn1, vn2, vn3, addr0, addr1, addr2, addr3) \
    vldp_post(vd0, vn0, addr0); \
    vldp_post(vd1, vn1, addr1); \
    vldp_post(vd2, vn2, addr2); \
    vldp_post(vd3, vn3, addr3);

#define vstr_m4(vd0, vd1, vd2, vd3, addr, off0, off1, off2, off3) \
    vstr(vd0, addr, off0); \
    vstr(vd1, addr, off1); \
    vstr(vd2, addr, off2); \
    vstr(vd3, addr, off3);

#define vzero(vd) \
    asm volatile("eor v" #vd ".16b, v" #vd ".16b, v" #vd ".16b" : : : "v" #vd)

#define vzero_m2(vd0, vd1) \
    vzero(vd0); \
    vzero(vd1);

#define vzero_m4(vd0, vd1, vd2, vd3) \
    vzero_m2(vd0, vd1); \
    vzero_m2(vd2, vd3);

#define vzero_m8(vd0, vd1, vd2, vd3, vd4, vd5, vd6, vd7) \
    vzero_m4(vd0, vd1, vd2, vd3); \
    vzero_m4(vd4, vd5, vd6, vd7);

#define vfadd(vd, vn, vm) \
    asm volatile("fadd v" #vd ".4s, v" #vn ".4s, v" #vm ".4s" : : : "v" #vd)

#define vfadd_m4(vd0, vd1, vd2, vd3, vn0, vn1, vn2, vn3, vm0, vm1, vm2, vm3) \
    vfadd(vd0, vn0, vm0); \
    vfadd(vd1, vn1, vm1); \
    vfadd(vd2, vn2, vm2); \
    vfadd(vd3, vn3, vm3);

#define vfmla(vd, vn, vm, lane) \
    asm volatile("fmla v" #vd ".4s, v" #vn ".4s, v" #vm ".s[" #lane "]" : : : "v" #vd)

#define vfmla_m2(vd0, vd1, vn, vm0, vm1, lane) \
    vfmla(vd0, vn, vm0, lane); \
    vfmla(vd1, vn, vm1, lane);

#define vfmla_m3(vd0, vd1, vd2, vn, vm0, vm1, vm2, lane) \
    vfmla(vd0, vn, vm0, lane); \
    vfmla(vd1, vn, vm1, lane); \
    vfmla(vd2, vn, vm2, lane);

#define vfmla_m4(vd0, vd1, vd2, vd3, vn, vm0, vm1, vm2, vm3, lane) \
    vfmla(vd0, vn, vm0, lane); \
    vfmla(vd1, vn, vm1, lane); \
    vfmla(vd2, vn, vm2, lane); \
    vfmla(vd3, vn, vm3, lane);

#define vfmla_m4_pa(vd0, vd1, vd2, vd3, vn, vm) \
    vfmla(vd0, vn, vm, 0); \
    vfmla(vd1, vn, vm, 1); \
    vfmla(vd2, vn, vm, 2); \
    vfmla(vd3, vn, vm, 3);

#define vfmla_m4_n4_ldr(vd0, vd1, vd2, vd3, vn0, vn1, vn2, vn3, vm0, vm1, vm2, vm3, lane, baddr) \
    vfmla_m4(vd0, vd1, vd2, vd3, vn0, vm0, vm1, vm2, vm3, lane); \
    vldr(vn0, baddr, 0); \
    vfmla_m4(vd0, vd1, vd2, vd3, vn1, vm0, vm1, vm2, vm3, lane); \
    vldr(vn1, baddr, 16); \
    vfmla_m4(vd0, vd1, vd2, vd3, vn2, vm0, vm1, vm2, vm3, lane); \
    vldr(vn2, baddr, 32); \
    vfmla_m4(vd0, vd1, vd2, vd3, vn3, vm0, vm1, vm2, vm3, lane); \
    vldr(vn3, baddr, 48); 

#define vfmla_m4_n4(vd0, vd1, vd2, vd3, vn0, vn1, vn2, vn3, vm0, vm1, vm2, vm3, lane) \
    vfmla_m4(vd0, vd1, vd2, vd3, vn0, vm0, vm1, vm2, vm3, lane); \
    vfmla_m4(vd0, vd1, vd2, vd3, vn1, vm0, vm1, vm2, vm3, lane); \
    vfmla_m4(vd0, vd1, vd2, vd3, vn2, vm0, vm1, vm2, vm3, lane); \
    vfmla_m4(vd0, vd1, vd2, vd3, vn3, vm0, vm1, vm2, vm3, lane);

#define vfmul(vd, vn, vm, lane) \
    asm volatile("fmul v" #vd ".4s, v" #vn ".4s, v" #vm ".s[" #lane "]" : : : "memory", "v" #vd)

#define vfmla_vec(vd, vn, vm) \
    asm volatile("fmla v" #vd ".4s, v" #vn ".4s, v" #vm ".4s" : : : "v" #vd)

#define vhsum(vd) \
    asm volatile("faddp v" #vd ".4s, v" #vd ".4s, v" #vd ".4s" : : : "v" #vd); \
    asm volatile("faddp v" #vd ".4s, v" #vd ".4s, v" #vd ".4s" : : : "v" #vd)

#define vst1s(vd, lane, addr) \
    asm volatile("st1 {v" #vd ".s}[" #lane "], [%0]" : : "r"(addr) : "memory")

#define vld1s(vd, lane, addr) \
    asm volatile("ld1 {v" #vd ".s}[" #lane "], [%0]" : : "r"(addr) : "memory", "v" #vd)

#define vfadd_f32(vd, vn) \
    asm volatile("fadd s" #vd ", s" #vd ", s" #vn : : : "v" #vd)

#define vzip1_4s(vd, vn, vm) \
    asm volatile("zip1 v" #vd ".4s, v" #vn ".4s, v" #vm ".4s" : : : "v" #vd)

#define vzip2_4s(vd, vn, vm) \
    asm volatile("zip2 v" #vd ".4s, v" #vn ".4s, v" #vm ".4s" : : : "v" #vd)

#define vmov_s0(vd, vs, lane) \
    asm volatile("mov s" #vd " .s[0], v" #vs ".s[" #lane "]" : : : "v" #vd)

#define vins_s1(vd, vs, lane) \
    asm volatile("ins v" #vd ".s[1], v" #vs ".s[" #lane "]" : : : "v" #vd)


static inline void micro_kernel_4x16_nn(float *a, float *b, float *c, long int lda, long int ldb, long int ldc, long int k, long int c_state)
{
    float *a1 = a + lda;
    float *a2 = a + (lda << 1);
    float *a3 = a1 + (lda << 1);
    float *b1 = b + ldb;
    float *b2 = b + (ldb << 1);
    float *b3 = b1 + (ldb << 1);
    vzero_m8(16, 17, 18, 19, 20, 21, 22, 23);
    vzero_m8(24, 25, 26, 27, 28, 29, 30, 31);

    if (k >= 8) {
        k -= 8;
        vldp_post_m4(0, 1, 2, 3, 4, 5, 6, 7, a, a1, a2, a3);
        vldp(8, 9, b, 0); vldp(10, 11, b, 32);
        b = b + (ldb << 2);
        vldp(12, 13, b1, 0); vldp(14, 15, b1, 32);
        b1 = b1 + (ldb << 2);
        while (k >= 8) {
            k -= 8;
            vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0); vldr(8, b2, 0);
            vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0); vldr(9, b2, 16);
            vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0); vldr(10, b2, 32);
            vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0); vldr(11, b2, 48);
            b2 = b2 + (ldb << 2);

            vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 1); vldr(12, b3, 0);
            vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 1); vldr(13, b3, 16);
            vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 1); vldr(14, b3, 32);
            vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 1); vldr(15, b3, 48);
            b3 = b3 + (ldb << 2);

            vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 2); vldr(8, b, 0);
            vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 2); vldr(9, b, 16);
            vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 2); vldr(10, b, 32);
            vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 2); vldr(11, b, 48);
            b = b + (ldb << 2);

            vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 3); vldr(12, b1, 0);
            vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 3); vldr(13, b1, 16);
            vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 3); vldr(14, b1, 32);
            vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 3);
            vldr_post(0, a); vldr_post(1, a1); vldr_post(2, a2); vldr_post(3, a3);
            vldr(15, b1, 48);
            b1 = b1 + (ldb << 2);

            vfmla_m4(16, 20, 24, 28, 8, 4, 5, 6, 7, 0); vldr(8, b2, 0);
            vfmla_m4(17, 21, 25, 29, 9, 4, 5, 6, 7, 0); vldr(9, b2, 16);
            vfmla_m4(18, 22, 26, 30, 10, 4, 5, 6, 7, 0); vldr(10, b2, 32);
            vfmla_m4(19, 23, 27, 31, 11, 4, 5, 6, 7, 0); vldr(11, b2, 48);
            b2 = b2 + (ldb << 2);

            vfmla_m4(16, 20, 24, 28, 12, 4, 5, 6, 7, 1); vldr(12, b3, 0);
            vfmla_m4(17, 21, 25, 29, 13, 4, 5, 6, 7, 1); vldr(13, b3, 16);
            vfmla_m4(18, 22, 26, 30, 14, 4, 5, 6, 7, 1); vldr(14, b3, 32);
            vfmla_m4(19, 23, 27, 31, 15, 4, 5, 6, 7, 1); vldr(15, b3, 48);
            b3 = b3 + (ldb << 2);

            vfmla_m4(16, 20, 24, 28, 8, 4, 5, 6, 7, 2); vldr(8, b, 0);
            vfmla_m4(17, 21, 25, 29, 9, 4, 5, 6, 7, 2); vldr(9, b, 16);
            vfmla_m4(18, 22, 26, 30, 10, 4, 5, 6, 7, 2); vldr(10, b, 32);
            vfmla_m4(19, 23, 27, 31, 11, 4, 5, 6, 7, 2); vldr(11, b, 48);
            b = b + (ldb << 2);

            vfmla_m4(16, 20, 24, 28, 12, 4, 5, 6, 7, 3); vldr(12, b1, 0);
            vfmla_m4(17, 21, 25, 29, 13, 4, 5, 6, 7, 3); vldr(13, b1, 16);
            vfmla_m4(18, 22, 26, 30, 14, 4, 5, 6, 7, 3); vldr(14, b1, 32);
            vfmla_m4(19, 23, 27, 31, 15, 4, 5, 6, 7, 3);
            vldr_post(4, a); vldr_post(5, a1); vldr_post(6, a2); vldr_post(7, a3);
            vldr(15, b1, 48);
            b1 = b1 + (ldb << 2);
        }
        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0); vldr(8, b2, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0); vldr(9, b2, 16);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0); vldr(10, b2, 32);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0); vldr(11, b2, 48);
        b2 = b2 + (ldb << 2);

        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 1); vldr(12, b3, 0);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 1); vldr(13, b3, 16);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 1); vldr(14, b3, 32);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 1); vldr(15, b3, 48);
        b3 = b3 + (ldb << 2);

        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 2); vldr(8, b, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 2); vldr(9, b, 16);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 2); vldr(10, b, 32);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 2); vldr(11, b, 48);
        b = b + (ldb << 2);

        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 3); vldr(12, b1, 0);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 3); vldr(13, b1, 16);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 3); vldr(14, b1, 32);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 3); vldr(15, b1, 48);
        b1 = b1 + (ldb << 2);

        vfmla_m4(16, 20, 24, 28, 8, 4, 5, 6, 7, 0); vldr(8, b2, 0);
        vfmla_m4(17, 21, 25, 29, 9, 4, 5, 6, 7, 0); vldr(9, b2, 16);
        vfmla_m4(18, 22, 26, 30, 10, 4, 5, 6, 7, 0); vldr(10, b2, 32);
        vfmla_m4(19, 23, 27, 31, 11, 4, 5, 6, 7, 0); vldr(11, b2, 48);
        b2 = b2 + (ldb << 2);

        vfmla_m4(16, 20, 24, 28, 12, 4, 5, 6, 7, 1); vldr(12, b3, 0);
        vfmla_m4(17, 21, 25, 29, 13, 4, 5, 6, 7, 1); vldr(13, b3, 16);
        vfmla_m4(18, 22, 26, 30, 14, 4, 5, 6, 7, 1); vldr(14, b3, 32);
        vfmla_m4(19, 23, 27, 31, 15, 4, 5, 6, 7, 1); vldr(15, b3, 48);
        b3 = b3 + (ldb << 2);

        vfmla_m4(16, 20, 24, 28, 8, 4, 5, 6, 7, 2);
        vfmla_m4(17, 21, 25, 29, 9, 4, 5, 6, 7, 2);
        vfmla_m4(18, 22, 26, 30, 10, 4, 5, 6, 7, 2);
        vfmla_m4(19, 23, 27, 31, 11, 4, 5, 6, 7, 2);

        vfmla_m4(16, 20, 24, 28, 12, 4, 5, 6, 7, 3);
        vfmla_m4(17, 21, 25, 29, 13, 4, 5, 6, 7, 3);
        vfmla_m4(18, 22, 26, 30, 14, 4, 5, 6, 7, 3);
        vfmla_m4(19, 23, 27, 31, 15, 4, 5, 6, 7, 3);
    }

    if (k >= 4) {
        k -= 4;
        vldr_post_m4(0, 1, 2, 3, a, a1, a2, a3);
        vldp(8, 9, b, 0); vldp(10, 11, b, 32);
        b = b + (ldb << 2);
        vldp(12, 13, b1, 0); vldp(14, 15, b1, 32);
        b1 = b1 + (ldb << 2);

        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0); vldr(8, b2, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0); vldr(9, b2, 16);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0); vldr(10, b2, 32);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0); vldr(11, b2, 48);
        b2 = b2 + (ldb << 2);

        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 1); vldr(12, b3, 0);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 1); vldr(13, b3, 16);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 1); vldr(14, b3, 32);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 1); vldr(15, b3, 48);
        b3 = b3 + (ldb << 2);

        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 2);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 2);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 2);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 2);

        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 3);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 3);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 3);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 3);
    }

    if (k >= 2) {
        k -= 2;
        vldr_f64_post_m4(0, 1, 2, 3, a, a1, a2, a3);
        vldp(8, 9, b, 0); vldp(10, 11, b, 32);
        b = b + (ldb << 1);
        vldp(12, 13, b1, 0); vldp(14, 15, b1, 32);
        b1 = b1 + (ldb << 1);

        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0);
        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 1);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 1);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 1);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 1);
    }

    if (k >= 1) {
        k -= 1;
        vldr_f32_post_m4(0, 1, 2, 3, a, a1, a2, a3);
        vldp(8, 9, b, 0); vldp(10, 11, b, 32);
        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0);
    }

    float *c1 = c + ldc;
    float *c2 = c + (ldc << 1);
    float *c3 = c1 + (ldc << 1);
    if (c_state) {
        vldr(0, c, 0);  vfadd(16, 16, 0); vstr(16, c, 0);
        vldr(1, c, 16); vfadd(17, 17, 1); vstr(17, c, 16);
        vldr(2, c, 32); vfadd(18, 18, 2); vstr(18, c, 32);
        vldr(3, c, 48); vfadd(19, 19, 3); vstr(19, c, 48);
        vldr(4, c1, 0); vfadd(20, 20, 4); vstr(20, c1, 0);
        vldr(5, c1, 16);vfadd(21, 21, 5); vstr(21, c1, 16);
        vldr(6, c1, 32);vfadd(22, 22, 6); vstr(22, c1, 32);
        vldr(7, c1, 48);vfadd(23, 23, 7); vstr(23, c1, 48);
        vldr(8, c2, 0); vfadd(24, 24, 8); vstr(24, c2, 0);
        vldr(9, c2, 16);vfadd(25, 25, 9); vstr(25, c2, 16);
        vldr(10,c2, 32);vfadd(26, 26, 10);vstr(26, c2, 32);
        vldr(11,c2, 48);vfadd(27, 27, 11);vstr(27, c2, 48);
        vldr(12,c3, 0); vfadd(28, 28, 12);vstr(28, c3, 0);
        vldr(13,c3, 16);vfadd(29, 29, 13);vstr(29, c3, 16);
        vldr(14,c3, 32);vfadd(30, 30, 14);vstr(30, c3, 32);
        vldr(15,c3, 48);vfadd(31, 31, 15);vstr(31, c3, 48);
    } else {
        vstr(16, c, 0); vstr(17, c, 16); vstr(18, c, 32); vstr(19, c, 48);
        vstr(20, c1, 0);vstr(21, c1, 16);vstr(22, c1, 32);vstr(23, c1, 48);
        vstr(24, c2, 0);vstr(25, c2, 16);vstr(26, c2, 32);vstr(27, c2, 48);
        vstr(28, c3, 0);vstr(29, c3, 16);vstr(30, c3, 32);vstr(31, c3, 48);
    }
}

static inline void micro_kernel_4x16_nn_pbg(float *a, float *b, float *c, long int lda, long int ldb, long int ldc, long int k, long int c_state, float *pb)
{
    float *a1 = a + lda;
    float *a2 = a + (lda << 1);
    float *a3 = a1 + (lda << 1);
    float *b1 = b + ldb;
    float *b2 = b + (ldb << 1);
    float *b3 = b1 + (ldb << 1);
    vzero_m8(16, 17, 18, 19, 20, 21, 22, 23);
    vzero_m8(24, 25, 26, 27, 28, 29, 30, 31);

    if (k >= 8) {
        k -= 8;
        vldp_post_m4(0, 1, 2, 3, 4, 5, 6, 7, a, a1, a2, a3);
        vldp(8, 9, b, 0); vldp(10, 11, b, 32);
        b = b + (ldb << 2);
        vstr_m4(8, 9, 10, 11, pb, 0, 16, 32, 48);
        pb += 16;
        vldp(12, 13, b1, 0); vldp(14, 15, b1, 32);
        b1 = b1 + (ldb << 2);
        vstr_m4(12, 13, 14, 15, pb, 0, 16, 32, 48);
        pb += 16;
        while (k >= 8) {
            k -= 8;
            vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0); vldr(8, b2, 0); vstr(8, pb, 0);
            vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0); vldr(9, b2, 16); vstr(9, pb, 16);
            vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0); vldr(10, b2, 32); vstr(10, pb, 32);
            vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0); vldr(11, b2, 48); vstr(11, pb, 48);
            b2 = b2 + (ldb << 2); pb += 16;

            vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 1); vldr(12, b3, 0); vstr(12, pb, 0);
            vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 1); vldr(13, b3, 16); vstr(13, pb, 16);
            vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 1); vldr(14, b3, 32); vstr(14, pb, 32);
            vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 1); vldr(15, b3, 48); vstr(15, pb, 48);
            b3 = b3 + (ldb << 2); pb += 16;

            vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 2); vldr(8, b, 0); vstr(8, pb, 0);
            vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 2); vldr(9, b, 16); vstr(9, pb, 16);
            vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 2); vldr(10, b, 32); vstr(10, pb, 32);
            vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 2); vldr(11, b, 48); vstr(11, pb, 48);
            b = b + (ldb << 2); pb += 16;

            vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 3); vldr(12, b1, 0); vstr(12, pb, 0);
            vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 3); vldr(13, b1, 16); vstr(13, pb, 16);
            vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 3); vldr(14, b1, 32); vstr(14, pb, 32);
            vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 3);
            vldr_post(0, a); vldr_post(1, a1); vldr_post(2, a2); vldr_post(3, a3);
            vldr(15, b1, 48); vstr(15, pb, 48);
            b1 = b1 + (ldb << 2); pb += 16;

            vfmla_m4(16, 20, 24, 28, 8, 4, 5, 6, 7, 0); vldr(8, b2, 0); vstr(8, pb, 0);
            vfmla_m4(17, 21, 25, 29, 9, 4, 5, 6, 7, 0); vldr(9, b2, 16); vstr(9, pb, 16);
            vfmla_m4(18, 22, 26, 30, 10, 4, 5, 6, 7, 0); vldr(10, b2, 32); vstr(10, pb, 32);
            vfmla_m4(19, 23, 27, 31, 11, 4, 5, 6, 7, 0); vldr(11, b2, 48); vstr(11, pb, 48);
            b2 = b2 + (ldb << 2); pb += 16;

            vfmla_m4(16, 20, 24, 28, 12, 4, 5, 6, 7, 1); vldr(12, b3, 0); vstr(12, pb, 0);
            vfmla_m4(17, 21, 25, 29, 13, 4, 5, 6, 7, 1); vldr(13, b3, 16); vstr(13, pb, 16);
            vfmla_m4(18, 22, 26, 30, 14, 4, 5, 6, 7, 1); vldr(14, b3, 32); vstr(14, pb, 32);
            vfmla_m4(19, 23, 27, 31, 15, 4, 5, 6, 7, 1); vldr(15, b3, 48); vstr(15, pb, 48);
            b3 = b3 + (ldb << 2); pb += 16;

            vfmla_m4(16, 20, 24, 28, 8, 4, 5, 6, 7, 2); vldr(8, b, 0); vstr(8, pb, 0);
            vfmla_m4(17, 21, 25, 29, 9, 4, 5, 6, 7, 2); vldr(9, b, 16); vstr(9, pb, 16);
            vfmla_m4(18, 22, 26, 30, 10, 4, 5, 6, 7, 2); vldr(10, b, 32); vstr(10, pb, 32);
            vfmla_m4(19, 23, 27, 31, 11, 4, 5, 6, 7, 2); vldr(11, b, 48); vstr(11, pb, 48);
            b = b + (ldb << 2); pb += 16;

            vfmla_m4(16, 20, 24, 28, 12, 4, 5, 6, 7, 3); vldr(12, b1, 0); vstr(12, pb, 0);
            vfmla_m4(17, 21, 25, 29, 13, 4, 5, 6, 7, 3); vldr(13, b1, 16); vstr(13, pb, 16);
            vfmla_m4(18, 22, 26, 30, 14, 4, 5, 6, 7, 3); vldr(14, b1, 32); vstr(14, pb, 32);
            vfmla_m4(19, 23, 27, 31, 15, 4, 5, 6, 7, 3);
            vldr_post(4, a); vldr_post(5, a1); vldr_post(6, a2); vldr_post(7, a3);
            vldr(15, b1, 48); vstr(15, pb, 48);
            b1 = b1 + (ldb << 2); pb += 16;
        }
        // Epilogue
        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0); vldr(8, b2, 0); vstr(8, pb, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0); vldr(9, b2, 16); vstr(9, pb, 16);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0); vldr(10, b2, 32); vstr(10, pb, 32);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0); vldr(11, b2, 48); vstr(11, pb, 48);
        b2 = b2 + (ldb << 2); pb += 16;

        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 1); vldr(12, b3, 0); vstr(12, pb, 0);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 1); vldr(13, b3, 16); vstr(13, pb, 16);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 1); vldr(14, b3, 32); vstr(14, pb, 32);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 1); vldr(15, b3, 48); vstr(15, pb, 48);
        b3 = b3 + (ldb << 2); pb += 16;

        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 2); vldr(8, b, 0); vstr(8, pb, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 2); vldr(9, b, 16); vstr(9, pb, 16);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 2); vldr(10, b, 32); vstr(10, pb, 32);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 2); vldr(11, b, 48); vstr(11, pb, 48);
        b = b + (ldb << 2); pb += 16;

        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 3); vldr(12, b1, 0); vstr(12, pb, 0);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 3); vldr(13, b1, 16); vstr(13, pb, 16);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 3); vldr(14, b1, 32); vstr(14, pb, 32);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 3); vldr(15, b1, 48); vstr(15, pb, 48);
        b1 = b1 + (ldb << 2); pb += 16;

        vfmla_m4(16, 20, 24, 28, 8, 4, 5, 6, 7, 0); vldr(8, b2, 0); vstr(8, pb, 0);
        vfmla_m4(17, 21, 25, 29, 9, 4, 5, 6, 7, 0); vldr(9, b2, 16); vstr(9, pb, 16);
        vfmla_m4(18, 22, 26, 30, 10, 4, 5, 6, 7, 0); vldr(10, b2, 32); vstr(10, pb, 32);
        vfmla_m4(19, 23, 27, 31, 11, 4, 5, 6, 7, 0); vldr(11, b2, 48); vstr(11, pb, 48);
        b2 = b2 + (ldb << 2); pb += 16;

        vfmla_m4(16, 20, 24, 28, 12, 4, 5, 6, 7, 1); vldr(12, b3, 0); vstr(12, pb, 0);
        vfmla_m4(17, 21, 25, 29, 13, 4, 5, 6, 7, 1); vldr(13, b3, 16); vstr(13, pb, 16);
        vfmla_m4(18, 22, 26, 30, 14, 4, 5, 6, 7, 1); vldr(14, b3, 32); vstr(14, pb, 32);
        vfmla_m4(19, 23, 27, 31, 15, 4, 5, 6, 7, 1); vldr(15, b3, 48); vstr(15, pb, 48);
        b3 = b3 + (ldb << 2); pb += 16;

        vfmla_m4(16, 20, 24, 28, 8, 4, 5, 6, 7, 2);
        vfmla_m4(17, 21, 25, 29, 9, 4, 5, 6, 7, 2);
        vfmla_m4(18, 22, 26, 30, 10, 4, 5, 6, 7, 2);
        vfmla_m4(19, 23, 27, 31, 11, 4, 5, 6, 7, 2);

        vfmla_m4(16, 20, 24, 28, 12, 4, 5, 6, 7, 3);
        vfmla_m4(17, 21, 25, 29, 13, 4, 5, 6, 7, 3);
        vfmla_m4(18, 22, 26, 30, 14, 4, 5, 6, 7, 3);
        vfmla_m4(19, 23, 27, 31, 15, 4, 5, 6, 7, 3);
    }

    if (k >= 4) {
        k -= 4;
        vldr_post_m4(0, 1, 2, 3, a, a1, a2, a3);
        vldp(8, 9, b, 0); vldp(10, 11, b, 32);
        b = b + (ldb << 2);
        vstr_m4(8, 9, 10, 11, pb, 0, 16, 32, 48);
        pb += 16;
        vldp(12, 13, b1, 0); vldp(14, 15, b1, 32);
        b1 = b1 + (ldb << 2);
        vstr_m4(12, 13, 14, 15, pb, 0, 16, 32, 48);
        pb += 16;

        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0); vldr(8, b2, 0); vstr(8, pb, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0); vldr(9, b2, 16); vstr(9, pb, 16);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0); vldr(10, b2, 32); vstr(10, pb, 32);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0); vldr(11, b2, 48); vstr(11, pb, 48);
        b2 = b2 + (ldb << 2); pb += 16;

        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 1); vldr(12, b3, 0); vstr(12, pb, 0);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 1); vldr(13, b3, 16); vstr(13, pb, 16);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 1); vldr(14, b3, 32); vstr(14, pb, 32);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 1); vldr(15, b3, 48); vstr(15, pb, 48);
        b3 = b3 + (ldb << 2); pb += 16;

        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 2);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 2);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 2);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 2);

        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 3);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 3);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 3);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 3);
    }

    if (k >= 2) {
        k -= 2;
        vldr_f64_post_m4(0, 1, 2, 3, a, a1, a2, a3);
        vldp(8, 9, b, 0); vldp(10, 11, b, 32);
        b = b + (ldb << 1);
        vstr_m4(8, 9, 10, 11, pb, 0, 16, 32, 48);
        pb += 16;
        vldp(12, 13, b1, 0); vldp(14, 15, b1, 32);
        b1 = b1 + (ldb << 1);
        vstr_m4(12, 13, 14, 15, pb, 0, 16, 32, 48);
        pb += 16;

        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0);
        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 1);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 1);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 1);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 1);
    }

    if (k >= 1) {
        k -= 1;
        vldr_f32_post_m4(0, 1, 2, 3, a, a1, a2, a3);
        vldp(8, 9, b, 0); vldp(10, 11, b, 32);
        vstr_m4(8, 9, 10, 11, pb, 0, 16, 32, 48);
        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0);
    }

    float *c1 = c + ldc;
    float *c2 = c + (ldc << 1);
    float *c3 = c1 + (ldc << 1);
    if (c_state) {
        vldr(0, c, 0);  vfadd(16, 16, 0); vstr(16, c, 0);
        vldr(1, c, 16); vfadd(17, 17, 1); vstr(17, c, 16);
        vldr(2, c, 32); vfadd(18, 18, 2); vstr(18, c, 32);
        vldr(3, c, 48); vfadd(19, 19, 3); vstr(19, c, 48);
        vldr(4, c1, 0); vfadd(20, 20, 4); vstr(20, c1, 0);
        vldr(5, c1, 16);vfadd(21, 21, 5); vstr(21, c1, 16);
        vldr(6, c1, 32);vfadd(22, 22, 6); vstr(22, c1, 32);
        vldr(7, c1, 48);vfadd(23, 23, 7); vstr(23, c1, 48);
        vldr(8, c2, 0); vfadd(24, 24, 8); vstr(24, c2, 0);
        vldr(9, c2, 16);vfadd(25, 25, 9); vstr(25, c2, 16);
        vldr(10,c2, 32);vfadd(26, 26, 10);vstr(26, c2, 32);
        vldr(11,c2, 48);vfadd(27, 27, 11);vstr(27, c2, 48);
        vldr(12,c3, 0); vfadd(28, 28, 12);vstr(28, c3, 0);
        vldr(13,c3, 16);vfadd(29, 29, 13);vstr(29, c3, 16);
        vldr(14,c3, 32);vfadd(30, 30, 14);vstr(30, c3, 32);
        vldr(15,c3, 48);vfadd(31, 31, 15);vstr(31, c3, 48);
    } else {
        vstr(16, c, 0); vstr(17, c, 16); vstr(18, c, 32); vstr(19, c, 48);
        vstr(20, c1, 0);vstr(21, c1, 16);vstr(22, c1, 32);vstr(23, c1, 48);
        vstr(24, c2, 0);vstr(25, c2, 16);vstr(26, c2, 32);vstr(27, c2, 48);
        vstr(28, c3, 0);vstr(29, c3, 16);vstr(30, c3, 32);vstr(31, c3, 48);
    }
}

static inline void micro_kernel_4x16_nn_pb(float *a, float *b, float *c, long int lda, long int ldb, long int ldc, long int k, long int c_state, float *pb)
{
    (void)b;
    (void)ldb;
    float *a1 = a + lda;
    float *a2 = a + (lda << 1);
    float *a3 = a1 + (lda << 1);
    vzero_m8(16, 17, 18, 19, 20, 21, 22, 23);
    vzero_m8(24, 25, 26, 27, 28, 29, 30, 31);

    if (k >= 8) {
        k -= 8;
        vldp_post_m4(0, 1, 2, 3, 4, 5, 6, 7, a, a1, a2, a3);
        vldp(8, 9, pb, 0); vldp(10, 11, pb, 32);
        pb += 16;
        vldp(12, 13, pb, 0); vldp(14, 15, pb, 32);
        pb += 16;
        while (k >= 8) {
            k -= 8;
            vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0); vldr(8, pb, 0);
            vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0); vldr(9, pb, 16);
            vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0); vldr(10, pb, 32);
            vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0); vldr(11, pb, 48);
            pb += 16;

            vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 1); vldr(12, pb, 0);
            vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 1); vldr(13, pb, 16);
            vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 1); vldr(14, pb, 32);
            vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 1); vldr(15, pb, 48);
            pb += 16;

            vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 2); vldr(8, pb, 0);
            vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 2); vldr(9, pb, 16);
            vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 2); vldr(10, pb, 32);
            vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 2); vldr(11, pb, 48);
            pb += 16;

            vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 3); vldr(12, pb, 0);
            vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 3); vldr(13, pb, 16);
            vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 3); vldr(14, pb, 32);
            vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 3);
            vldr_post(0, a); vldr_post(1, a1); vldr_post(2, a2); vldr_post(3, a3);
            vldr(15, pb, 48);
            pb += 16;

            vfmla_m4(16, 20, 24, 28, 8, 4, 5, 6, 7, 0); vldr(8, pb, 0);
            vfmla_m4(17, 21, 25, 29, 9, 4, 5, 6, 7, 0); vldr(9, pb, 16);
            vfmla_m4(18, 22, 26, 30, 10, 4, 5, 6, 7, 0); vldr(10, pb, 32);
            vfmla_m4(19, 23, 27, 31, 11, 4, 5, 6, 7, 0); vldr(11, pb, 48);
            pb += 16;

            vfmla_m4(16, 20, 24, 28, 12, 4, 5, 6, 7, 1); vldr(12, pb, 0);
            vfmla_m4(17, 21, 25, 29, 13, 4, 5, 6, 7, 1); vldr(13, pb, 16);
            vfmla_m4(18, 22, 26, 30, 14, 4, 5, 6, 7, 1); vldr(14, pb, 32);
            vfmla_m4(19, 23, 27, 31, 15, 4, 5, 6, 7, 1); vldr(15, pb, 48);
            pb += 16;

            vfmla_m4(16, 20, 24, 28, 8, 4, 5, 6, 7, 2); vldr(8, pb, 0);
            vfmla_m4(17, 21, 25, 29, 9, 4, 5, 6, 7, 2); vldr(9, pb, 16);
            vfmla_m4(18, 22, 26, 30, 10, 4, 5, 6, 7, 2); vldr(10, pb, 32);
            vfmla_m4(19, 23, 27, 31, 11, 4, 5, 6, 7, 2); vldr(11, pb, 48);
            pb += 16;

            vfmla_m4(16, 20, 24, 28, 12, 4, 5, 6, 7, 3); vldr(12, pb, 0);
            vfmla_m4(17, 21, 25, 29, 13, 4, 5, 6, 7, 3); vldr(13, pb, 16);
            vfmla_m4(18, 22, 26, 30, 14, 4, 5, 6, 7, 3); vldr(14, pb, 32);
            vfmla_m4(19, 23, 27, 31, 15, 4, 5, 6, 7, 3);
            vldr_post(4, a); vldr_post(5, a1); vldr_post(6, a2); vldr_post(7, a3);
            vldr(15, pb, 48);
            pb += 16;
        }
        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0); vldr(8, pb, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0); vldr(9, pb, 16);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0); vldr(10, pb, 32);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0); vldr(11, pb, 48);
        pb += 16;

        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 1); vldr(12, pb, 0);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 1); vldr(13, pb, 16);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 1); vldr(14, pb, 32);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 1); vldr(15, pb, 48);
        pb += 16;

        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 2); vldr(8, pb, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 2); vldr(9, pb, 16);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 2); vldr(10, pb, 32);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 2); vldr(11, pb, 48);
        pb += 16;

        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 3); vldr(12, pb, 0);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 3); vldr(13, pb, 16);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 3); vldr(14, pb, 32);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 3); vldr(15, pb, 48);
        pb += 16;

        vfmla_m4(16, 20, 24, 28, 8, 4, 5, 6, 7, 0); vldr(8, pb, 0);
        vfmla_m4(17, 21, 25, 29, 9, 4, 5, 6, 7, 0); vldr(9, pb, 16);
        vfmla_m4(18, 22, 26, 30, 10, 4, 5, 6, 7, 0); vldr(10, pb, 32);
        vfmla_m4(19, 23, 27, 31, 11, 4, 5, 6, 7, 0); vldr(11, pb, 48);
        pb += 16;

        vfmla_m4(16, 20, 24, 28, 12, 4, 5, 6, 7, 1); vldr(12, pb, 0);
        vfmla_m4(17, 21, 25, 29, 13, 4, 5, 6, 7, 1); vldr(13, pb, 16);
        vfmla_m4(18, 22, 26, 30, 14, 4, 5, 6, 7, 1); vldr(14, pb, 32);
        vfmla_m4(19, 23, 27, 31, 15, 4, 5, 6, 7, 1); vldr(15, pb, 48);
        pb += 16;

        vfmla_m4(16, 20, 24, 28, 8, 4, 5, 6, 7, 2);
        vfmla_m4(17, 21, 25, 29, 9, 4, 5, 6, 7, 2);
        vfmla_m4(18, 22, 26, 30, 10, 4, 5, 6, 7, 2);
        vfmla_m4(19, 23, 27, 31, 11, 4, 5, 6, 7, 2);

        vfmla_m4(16, 20, 24, 28, 12, 4, 5, 6, 7, 3);
        vfmla_m4(17, 21, 25, 29, 13, 4, 5, 6, 7, 3);
        vfmla_m4(18, 22, 26, 30, 14, 4, 5, 6, 7, 3);
        vfmla_m4(19, 23, 27, 31, 15, 4, 5, 6, 7, 3);
    }

    if (k >= 4) {
        k -= 4;
        vldr_post_m4(0, 1, 2, 3, a, a1, a2, a3);
        vldp(8, 9, pb, 0); vldp(10, 11, pb, 32);
        pb += 16;
        vldp(12, 13, pb, 0); vldp(14, 15, pb, 32);
        pb += 16;

        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0); vldr(8, pb, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0); vldr(9, pb, 16);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0); vldr(10, pb, 32);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0); vldr(11, pb, 48);
        pb += 16;

        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 1); vldr(12, pb, 0);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 1); vldr(13, pb, 16);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 1); vldr(14, pb, 32);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 1); vldr(15, pb, 48);
        pb += 16;

        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 2);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 2);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 2);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 2);

        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 3);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 3);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 3);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 3);
    }

    if (k >= 2) {
        k -= 2;
        vldr_f64_post_m4(0, 1, 2, 3, a, a1, a2, a3);
        vldp(8, 9, pb, 0); vldp(10, 11, pb, 32);
        pb += 16;
        vldp(12, 13, pb, 0); vldp(14, 15, pb, 32);
        pb += 16;

        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0);
        vfmla_m4(16, 20, 24, 28, 12, 0, 1, 2, 3, 1);
        vfmla_m4(17, 21, 25, 29, 13, 0, 1, 2, 3, 1);
        vfmla_m4(18, 22, 26, 30, 14, 0, 1, 2, 3, 1);
        vfmla_m4(19, 23, 27, 31, 15, 0, 1, 2, 3, 1);
    }

    if (k >= 1) {
        k -= 1;
        vldr_f32_post_m4(0, 1, 2, 3, a, a1, a2, a3);
        vldp(8, 9, pb, 0); vldp(10, 11, pb, 32);
        vfmla_m4(16, 20, 24, 28, 8, 0, 1, 2, 3, 0);
        vfmla_m4(17, 21, 25, 29, 9, 0, 1, 2, 3, 0);
        vfmla_m4(18, 22, 26, 30, 10, 0, 1, 2, 3, 0);
        vfmla_m4(19, 23, 27, 31, 11, 0, 1, 2, 3, 0);
    }

    float *c1 = c + ldc;
    float *c2 = c + (ldc << 1);
    float *c3 = c1 + (ldc << 1);
    if (c_state) {
        vldr(0, c, 0);  vfadd(16, 16, 0); vstr(16, c, 0);
        vldr(1, c, 16); vfadd(17, 17, 1); vstr(17, c, 16);
        vldr(2, c, 32); vfadd(18, 18, 2); vstr(18, c, 32);
        vldr(3, c, 48); vfadd(19, 19, 3); vstr(19, c, 48);
        vldr(4, c1, 0); vfadd(20, 20, 4); vstr(20, c1, 0);
        vldr(5, c1, 16);vfadd(21, 21, 5); vstr(21, c1, 16);
        vldr(6, c1, 32);vfadd(22, 22, 6); vstr(22, c1, 32);
        vldr(7, c1, 48);vfadd(23, 23, 7); vstr(23, c1, 48);
        vldr(8, c2, 0); vfadd(24, 24, 8); vstr(24, c2, 0);
        vldr(9, c2, 16);vfadd(25, 25, 9); vstr(25, c2, 16);
        vldr(10,c2, 32);vfadd(26, 26, 10);vstr(26, c2, 32);
        vldr(11,c2, 48);vfadd(27, 27, 11);vstr(27, c2, 48);
        vldr(12,c3, 0); vfadd(28, 28, 12);vstr(28, c3, 0);
        vldr(13,c3, 16);vfadd(29, 29, 13);vstr(29, c3, 16);
        vldr(14,c3, 32);vfadd(30, 30, 14);vstr(30, c3, 32);
        vldr(15,c3, 48);vfadd(31, 31, 15);vstr(31, c3, 48);
    } else {
        vstr(16, c, 0); vstr(17, c, 16); vstr(18, c, 32); vstr(19, c, 48);
        vstr(20, c1, 0);vstr(21, c1, 16);vstr(22, c1, 32);vstr(23, c1, 48);
        vstr(24, c2, 0);vstr(25, c2, 16);vstr(26, c2, 32);vstr(27, c2, 48);
        vstr(28, c3, 0);vstr(29, c3, 16);vstr(30, c3, 32);vstr(31, c3, 48);
    }
}

#endif  // KGEMM_NEON_KERNEL_H
