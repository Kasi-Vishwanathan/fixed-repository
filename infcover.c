/* infcover.c -- test zlib's inflate routines with full code coverage
 * Copyright (C) 2011, 2016, 2024 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* to use, do: ./configure --cover && make cover */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "zlib.h"

/* get definition of internal structure so we can mess with it (see pull()),
   and so we can call inflate_trees() (see cover5()) */
#define ZLIB_INTERNAL
#include "inftrees.h"
#include "inflate.h"

#define local static

/* -- memory tracking routines -- */

/*
   These memory tracking routines are provided to zlib and track all of zlib's
   allocations and deallocations, check for LIFO operations, keep a current
   and high water mark of total bytes requested, optionally set a limit on the
   total memory that can be allocated, and when done check for memory leaks.

   They are used as follows:

   z_stream strm;
   mem_setup(&strm)         initialize and install memory tracking
   ...
   mem_used(&strm, "msg")  show current memory usage
   ...
   mem_done(&strm, "msg")  close memory tracking, check for leaks
 */

/* These items are separated out from the strm structure to permit Z_NULL. */
local struct mem_zone {
    void **ptr;         /* pointer to allocated data */
    size_t size;        /* requested size of allocation */
    struct mem_zone *next;  /* pointer to next allocation in list */
} *zone;

local size_t total;     /* total bytes allocated */
local size_t highwater; /* peak total allocated */
local size_t limit;     /* memory allocation limit */
local int error;        /* error status */

local void *mem_alloc(void *mem, unsigned items, unsigned size);
local void mem_free(void *mem, void *ptr);

local static void mem_setup(z_stream *strm) {
    strm->zalloc = mem_alloc;
    strm->zfree = mem_free;
    strm->opaque = Z_NULL;
    total = 0;
    highwater = 0;
    limit = 0;
    error = 0;
    zone = Z_NULL;
}

/* ... [rest of the code with similar fixes for other functions] ... */