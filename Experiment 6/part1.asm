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
;r4 = 1, r5 = 10, r6 = 100, r7 = 1000, r15 = timer

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


Main		mov.b	@r4,		&P1OUT
			mov.b	#08h,		&P2DIR
			clr		&P1OUT
			mov.b	@r5,		&P1OUT
			mov.b	#04h,		&P2DIR
			clr		&P1OUT
			mov.b	@r6,		&P1OUT
			mov.b	#02h,		&P2DIR
			clr		&P1OUT
			mov.b	@r7,		&P1OUT
			mov.b	#01h,		&P2DIR
			clr		&P1OUT
			jmp		Main

;						0			1		   2	     3			4			5		  6			  7			8		   9
arr			.byte	00111111b, 00000110b, 01011011b, 01001111b, 01100110b, 01101101b, 01111101b, 00000111b, 01111111b, 01101111b
arr_end

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
            
