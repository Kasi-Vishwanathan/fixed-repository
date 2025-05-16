/* enough.c -- determine the maximum size of inflate's Huffman code tables over
 * all possible valid and complete prefix codes, subject to a length limit.
 * Copyright (C) 2007, 2008, 2012, 2018, 2024 Mark Adler
 * Version 1.6  29 July 2024  Mark Adler
 */

/* Version history:
   1.0   3 Jan 2007  First version (derived from codecount.c version 1.4)
   1.1   4 Jan 2007  Use faster incremental table usage computation
                     Prune examine() search on previously visited states
   1.2   5 Jan 2007  Comments clean up
                     As inflate does, decrease root for short codes
                     Refuse cases where inflate would increase root
   1.3  17 Feb 2008  Add argument for initial root table size
                     Fix bug for initial root table size == max - 1
                     Use a macro to compute the history index
   1.4  18 Aug 2012  Avoid shifts more than bits in type (caused endless loop!)
                     Clean up comparisons of different types
                     Provide enough() to return maximum table size
   1.5   2 Jan 2024  Add header documentation
                     Include <stdio.h> for printf()
   1.6  29 Jul 2024  Convert K&R functions to C99 prototypes
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#define local static

/* Maximum number of symbols, given maximum static bits in block type 2 is 15 */
#define MAXSYM 328

/* Maximum allowed bit length for the codes */
#define MAXBITS 15

/* Largest context Mat Hammer can handle (32768 for 16-bit ints) */
#define MAXCODE (1U << 15)

/* Maximum table size for the Huffman codes */
#define MAXSIZE 4320

/* Type for counting table sizes */
typedef unsigned long size_t;

/* The examine() function is called for every possible Huffman code of length
   up to max, and for every allowed number of symbols. It determines the maximum
   table size used by zlib's inflate, which uses an initial root table followed
   by length-indexed tabels. Search all possibilities, using a history of
   states that have already been searched. */
local void examine(int sym, int left, int length, int sum, int used, int pre, int lls);

/* History of visited states, used to avoid redundant searches */
typedef struct {int sym, left, length;} state_t;
local state_t *history;
local size_t hist;

/* Keep track of maximum table usage found */
local int max;

/* These global variables are used to pass parameters from enough() to examine(). */
local int root, maxbits;

/* Calculate the maximum table size needed by inflate for a given number of
   symbols, minimum and maximum code lengths, and initial root table size.
   If the computed maximum exceeds the compression's subdivs parameter (default
   15), the compression will have to use an internal temporary workspace. */
local int enough(int syms, int min, int maxlen, int rootin) {
    // ... rest of the function remains unchanged ...
}

/* For a current code length length (<= maxbits), compute how many left code
   patterns there are, how many symbols remain to be coded, and compute the
   sum of symbol counts up to this length. Also, compute the maximum table size
   used so far (used), which is the sum of the root table and all subsequent
   tables needed. Track the maximal used value over all possible Huffman codes. */
local void examine(int sym, int left, int length, int sum, int used, int pre, int lls) {
    // ... rest of the function remains unchanged ...
}

/* Process the command line arguments and compute the maximum table size. */
int main(int argc, char **argv) {
    // ... rest of the function remains unchanged ...
}