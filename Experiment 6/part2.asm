;-------------------------------------------------------------------------------
; MSP430 Assembler Code Template for use with TI Code Composer Studio
;
;
;-------------------------------------------------------------------------------
            .cdecls C,LIST,"msp430.h"       ; Include device header file
            
;-------------------------------------------------------------------------------
            .def    RESET                   ; Export program entry-point to
                                            ; make it known to linker.
;-------------------------------------------------------------------------------
            .text                           ; Assemble into program memory.
            .retain                         ; Override ELF conditional linking
                                            ; and retain current section.
            .retainrefs                     ; And retain any sections that have
                                            ; references to current section.

;-------------------------------------------------------------------------------
RESET       mov.w   #__STACK_END,SP         ; Initialize stackpointer
StopWDT     mov.w   #WDTPW|WDTHOLD,&WDTCTL  ; Stop watchdog timer


;-------------------------------------------------------------------------------
; Main loop here
;-------------------------------------------------------------------------------


setup_INT   bis.b   #040h,      &P2IE       ;Enabling interrupt
            and.b   #0BFh,      &P2SEL      ; 0 = general purpose I/O function
            and.b   #0BFh,      &P2SEL2     ; 1 = peripheral module function
            bis.b   #040h,      &P2IES      ; High to low interrupt mode, only works when you let go of the button
            clr     &P2IFG ; Clearing flags
            eint    ;Enabling interrupt

;r4 = 1, r5 = 10, r6 = 100, r7 = 1000,
Setup		bis.b	#0FFh,		&P1DIR
			bis.b	#00Fh,		&P2DIR
			bic.b	#0FFh,		&P1OUT
			mov.b	#001h,		&P2OUT
			mov.w	#arr, 		r4
			add		#6,			r4
			mov.w	#arr, 		r5
			add		#4,			r5
			mov.w	#arr, 		r6
			add		#2,			r6
			mov.w	#arr, 		r7

Set_timer	; TA0CTL 15-10..100001x010
			; TA0CCR0	#10486d
			; TA0CCTL0  00??x?x00011x?x0
			mov.w	#01000010000b,	TA0CTL
			mov.w	#10486d,	TA0CCR0
			mov.w	#0000000000010000b,	TA0CCTL0

Main		call	#BCD2Dec
			mov.b	@r4,		&P1OUT
			mov.b	#08h,		&P2OUT
			nop
			nop
			clr		&P1OUT
			clr		&P2OUT
			mov.b	@r5,		&P1OUT
			mov.b	#04h,		&P2OUT
			nop
			nop
			clr		&P1OUT
			clr		&P2OUT
			mov.b	@r6,		&P1OUT
			mov.b	#02h,		&P2OUT
			nop
			nop
			clr		&P1OUT
			clr		&P2OUT
			mov.b	@r7,		&P1OUT
			mov.b	#01h,		&P2OUT
			nop
			nop
			clr		&P1OUT
			clr		&P2OUT
			jmp		Main

;-------------------------------------------------------------------------------
; Interrupt Service Routine for Resetting Timer
;-------------------------------------------------------------------------------

ISR			dint
			mov.b 	#00h,  		sec
			mov.b	#00h,		csec
			clr		&P2IFG
			eint
			reti

;-------------------------------------------------------------------------------
; Timer Interrut Service Routine
;-------------------------------------------------------------------------------

TISR		dint
			push	r15
			add.b	#1b,		csec
			mov.b	csec,		r15
			bic.b	#0F0h,		r15
			cmp		#0Ah,		r15
			jz		ADDDecSec
			jmp		TISRend

ADDDecSec	add.b	#010h,		csec
			bic.b	#00Fh,		csec
			mov.b	csec,		r15
			bic.b	#00Fh,		r15
			cmp		#0A0h,		r15
			jz		ADDSec
			jmp		TISRend

ADDSec		add.b	#001h,		sec
			bic.b	#0FFh,		csec
			mov.b	sec,		r15
			bic.b	#0F0h,		r15
			cmp		#0Ah,		r15
			jz		ADDDekSec
			jmp		TISRend

ADDDekSec	add.b	#010h,		sec
			bic.b	#00Fh,		sec
			mov.b	sec,		r15
			bic.b	#00Fh,		r15
			cmp		#0A0h,		r15
			jz		RESET

TISRend		pop 	r15
			eint
			reti

;-------------------------------------------------------------------------------
; Conversion to Decimal from BCD
;-------------------------------------------------------------------------------

BCD2Dec		push 	r14

			mov.b	csec,	r14
			bic.b	#0F0h,	r14
			mov.w	#arr,	r4
			add.w	r14,	r4

			mov.b	csec,	r14
			rra.b	r14
			rra.b	r14
			rra.b	r14
			rra.b	r14
			bic.b	#0F0h,	r14
			mov.w	#arr,	r5
			add.w	r14,	r5


			mov.b	sec,	r14
			bic.b	#0F0h,	r14
			mov.w	#arr,	r6
			add.w	r14,	r6

			mov.b	sec,	r14
			rra.b	r14
			rra.b	r14
			rra.b	r14
			rra.b	r14
			bic.b	#0F0h,	r14
			mov.w	#arr,	r7
			add.w	r14,	r7

			pop		r14
			ret

;						0			1		   2	     3			4			5		  6			  7			8		   9
arr			.byte	00111111b, 00000110b, 01011011b, 01001111b, 01100110b, 01101101b, 01111101b, 00000111b, 01111111b, 01101111b
arr_end
			.data
sec			.byte	00h
csec 		.byte 	00h

;-------------------------------------------------------------------------------
; Stack Pointer definition
;-------------------------------------------------------------------------------
            .global __STACK_END
            .sect   .stack
            
;-------------------------------------------------------------------------------
; Interrupt Vectors
;-------------------------------------------------------------------------------
            .sect   ".reset"                ; MSP430 RESET Vector
            .short  RESET
            .sect	".int03"
            .short  ISR
            .sect 	".int09"
            .short	TISR
