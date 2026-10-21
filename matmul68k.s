;-----------------------------------------------------------------------------
; matmul68k.s -- int8 dot product for the Amiga 500 (68000)
;
;   long dot_i8_x16(const signed char *w, const short *x, int n);
;
; w is a row of int8 weights, x is the activation vector ALREADY sign-extended
; to 16-bit words by the caller. Expanding x once per matmul instead of once
; per row saves a MOVE.B + EXT.W on every single multiply-accumulate.
;
; Standard m68k C calling convention: arguments on the stack, result in D0,
; D2-D7/A2-A6 preserved. Assemble with vasm:
;     vasmm68k_mot -Fhunk -m68000 -no-opt matmul68k.s -o matmul68k.o
;
; Cost per MAC, 68000 @ 7.09 MHz:
;     MOVE.B (A0)+,D1     8
;     EXT.W   D1          4
;     MULS.W (A1)+,D1    46 typical  (38 + 2n, n = bit transitions in source)
;                       + 8 for the (A1)+ operand fetch
;     ADD.L   D1,D0       8
;                      ----
;                        74 cycles, plus ~1.3 for the amortised DBRA
;-----------------------------------------------------------------------------

        XDEF    _dot_i8_x16

        SECTION code,CODE

_dot_i8_x16:
        movem.l d2-d3,-(sp)             ; 8 bytes saved
        move.l  12(sp),a0               ; w   (8 saved + 4 return address)
        move.l  16(sp),a1               ; x
        move.l  20(sp),d3               ; n
        moveq   #0,d0                   ; accumulator

        move.l  d3,d2
        lsr.l   #3,d2                   ; blocks of eight
        beq     .rem
        subq.w  #1,d2

.loop8:
        move.b  (a0)+,d1
        ext.w   d1
        muls.w  (a1)+,d1
        add.l   d1,d0

        move.b  (a0)+,d1
        ext.w   d1
        muls.w  (a1)+,d1
        add.l   d1,d0

        move.b  (a0)+,d1
        ext.w   d1
        muls.w  (a1)+,d1
        add.l   d1,d0

        move.b  (a0)+,d1
        ext.w   d1
        muls.w  (a1)+,d1
        add.l   d1,d0

        move.b  (a0)+,d1
        ext.w   d1
        muls.w  (a1)+,d1
        add.l   d1,d0

        move.b  (a0)+,d1
        ext.w   d1
        muls.w  (a1)+,d1
        add.l   d1,d0

        move.b  (a0)+,d1
        ext.w   d1
        muls.w  (a1)+,d1
        add.l   d1,d0

        move.b  (a0)+,d1
        ext.w   d1
        muls.w  (a1)+,d1
        add.l   d1,d0

        dbra    d2,.loop8

.rem:
        move.l  d3,d2
        and.w   #7,d2
        beq     .done
        subq.w  #1,d2
.loop1:
        move.b  (a0)+,d1
        ext.w   d1
        muls.w  (a1)+,d1
        add.l   d1,d0
        dbra    d2,.loop1

.done:
        movem.l (sp)+,d2-d3
        rts

        END
