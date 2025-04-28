/* adler32.c -- compute the Adler-32 checksum of a data stream
 * Copyright (C) 1995-2011, 2016 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */

#include "zutil.h"

#define BASE 65521U     /* largest prime smaller than 65536 */
#define NMAX 5552
/* NMAX is the largest n such that 255n(n+1/2) + (n+1)(BASE-1) <= 2^32-1 */

#define DO1(buf,i)  {adler += (buf)[i]; sum2 += adler;}
#define DO2(buf,i)  DO1(buf,i); DO1(buf,i+1);
#define DO4(buf,i)  DO2(buf,i); DO2(buf,i+2);
#define DO8(buf,i)  DO4(buf,i); DO4(buf,i+4);
#define DO16(buf)   DO8(buf,0); DO8(buf,8);

#ifdef NO_DIVIDE
#  define CHOP(a) \
    do { \
        unsigned long tmp = (a) >> 16; \
        (a) &= 0xffffUL; \
        (a) += (tmp << 4) - tmp; \
    } while (0)
#endif

/* ========================================================================= */
uLong ZEXPORT adler32(uLong adler, const Bytef *buf, uInt len) {
    unsigned long sum2;
    unsigned n;

    /* Split Adler-32 into component sums */
    sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* Handle short blocks */
    if (len == 1) {
        adler += buf[0];
        if (adler >= BASE)
            adler -= BASE;
        sum2 += adler;
        if (sum2 >= BASE)
            sum2 -= BASE;
        return adler | (sum2 << 16);
    }

    /* Initial Adler-32 value (deferred check for len == 1 speed) */
    if (buf == Z_NULL)
        return 1L;

    /* Process remainder */
    while (len >= NMAX) {
        len -= NMAX;
        n = NMAX;
        do {
            adler += *buf++;
            sum2 += adler;
        } while (--n);
        adler %= BASE;
        sum2 %= BASE;
    }
    if (len) {
        while (len >= 16) {
            len -= 16;
            DO16(buf);
            buf += 16;
        }
        while (len--) {
            adler += *buf++;
            sum2 += adler;
        }
        adler %= BASE;
        sum2 %= BASE;
    }

    /* Return recombined sums */
    return adler | (sum2 << 16);
}