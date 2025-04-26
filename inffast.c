/* inffast.c -- fast decoding
 * Copyright (C) 1995-2017 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zutil.h"
#include "inftrees.h"
#include "inflate.h"
#include "inffast.h"

#ifdef ASMINF
#  pragma message("Assembler code may have bugs -- use at your own risk")
#else

/*
   Decode literal, length, and distance codes and write out the resulting
   literal and match bytes until either not enough input or output is
   available, an end-of-block is encountered, or a data error is encountered.
   When large enough input and output buffers are supplied to inflate(), for
   example, a 16K input buffer and a 64K output buffer, more than 95% of the
   inflate execution time is spent in this routine.

   Entry assumptions:

        state->mode == LEN
        strm->avail_in >= 6
        strm->avail_out >= 258
        start >= strm->avail_out
        state->bits < 8

   On return, state->mode is one of:

        LEN -- ran out of enough outp
*/

/* Updated function definition to ANSI C style */
void inflate_fast(z_streamp strm, unsigned start) {
    struct inflate_state *state;
    unsigned char *in;      /* local strm->next_in */
    unsigned char *last;    /* while in < last, enough input available */
    unsigned char *out;     /* local strm->next_out */
    unsigned char *beg;     /* inflate()'s initial strm->next_out */
    unsigned char *end;     /* while out < end, enough space available */
#ifdef INFLATE_STRICT
    unsigned dmax;          /* maximum distance from zlib header */
#endif
    unsigned wsize;         /* window size or zero if not using window */
    unsigned whave;         /* valid bytes in the window */
    unsigned wnext;         /* window write index */
    unsigned char *window;  /* allocated sliding window, if wsize != 0 */
    unsigned long hold;     /* local strm->hold */
    unsigned bits;          /* local strm->bits */
    code const *lcode;      /* local strm->lencode */
    code const *dcode;      /* local strm->distcode */
    unsigned lmask;         /* mask for first level of length codes */
    unsigned dmask;         /* mask for first level of distance codes */
    code here;              /* retrieved table entry */
    unsigned op;            /* code bits, operation, extra bits, or */
                            /*  window position, window bytes to copy */
    unsigned len;           /* match length, unused bytes */
    unsigned dist;          /* match distance */
    unsigned char *from;    /* where to copy match from */

    /* ... rest of the function body remains unchanged ... */
}
#endif