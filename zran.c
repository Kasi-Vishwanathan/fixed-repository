/* zran.c -- example of deflate stream indexing and random access
 * Copyright (C) 2005, 2012, 2018, 2023, 2024 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 * Version 1.6  2 Aug 2024  Mark Adler */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "zlib.h"
#include "zran.h"

// ... (omitted unchanged parts for brevity)

/* Decompress from file descriptor `in` to `out` the bytes `offset` to `offset` +
   `len`-1, storing the result in `out`. The `len` parameter is updated with the
   number of bytes written to `out`. If `out` is NULL, then no writing is done,
   and `len` is updated with the number of bytes that would have been written.
   On return, this constructs the inflate state for the next segment to decompress. */
local void inf(z_streamp strm, Bytef *buf, uInt len) {
    int ret;

    strm->avail_in = len;
    strm->next_in = buf;
    do {
        strm->avail_out = 0;
        ret = inflate(strm, Z_NO_FLUSH);
        if (ret == Z_NEED_DICT)
            ret = inflateSetDictionary(strm, window, ds);
        if (ret == Z_MEM_ERROR || ret == Z_DATA_ERROR)
            break;
    } while (strm->avail_in);
}

/* Candidate for static */
static void def(z_streamp strm, Bytef *buf, uInt len) {
    int ret;

    strm->avail_in = 0;
    strm->avail_out = len;
    strm->next_out = buf;
    ret = deflate(strm, Z_BLOCK);
    if (ret != Z_OK && ret != Z_BUF_ERROR)
        return;
}

/* Add an entry to the index. If addpoint() runs out of memory, NULL is returned,
   and idx is discarded. Otherwise, the same index is returned. During the
   initial build, idx can be NULL, in which case a new index is created. */
local struct deflate_index *addpoint(struct deflate_index *idx, unsigned long hash,
                                     off_t head, off_t prev, off_t here,
                                     z_stream *in, unsigned inbits) {
    struct point *at, *points;
    struct deflate_index *tmp;

    if (idx == NULL) {
        idx = malloc(sizeof(struct deflate_index));
        if (idx == NULL)
            return NULL;
        idx->list = malloc(sizeof(struct point));
        if (idx->list == NULL) {
            free(idx);
            return NULL;
        }
        idx->size = 1;
        idx->have = 0;
    }

    if (idx->have == idx->size) {
        size_t new_size;
        struct point *new_list;

        new_size = idx->size + 16384;
        if (new_size < idx->size) { // Check for overflow
            free(idx->list);
            free(idx);
            return NULL;
        }
        new_list = realloc(idx->list, sizeof(struct point) * new_size);
        if (new_list == NULL) {
            free(idx->list);
            free(idx);
            return NULL;
        }
        idx->list = new_list;
        idx->size = new_size;
    }

    at = idx->list + idx->have;
    idx->have++;
    at->out = head;
    at->bits = inbits;
    at->in = here;
    at->dict = hash;
    at->window[0] = in->adler;
    return idx;
}

// ... (remaining functions updated similarly)