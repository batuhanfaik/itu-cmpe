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

;			Part 4
Setup3		mov.b	#0FFh,		&P1DIR
			bic.b	#0002h,		&P2DIR
			mov.b	#0006h,		&P2IE
			bic.b	#0006h,		&P2IES
			mov.b	#0000h,		&P2IFG
			mov.b	#0000h,		&P1OUT

Main		bit.b	#0002h,		&P2IFG
			jnz 	increment
			bit.b	#0004h,		&P2IFG
			jnz		complement
			jmp		Main

section 	.data
counter		.word	0000h

increment	inc.b	counter
			mov.b	counter,	&P1OUT
			bic.b	#0002h,		&P2IFG
			jmp 	Main

complement	xor.b	#0FFh,		counter
			inc.b	counter
			mov.b	counter,	&P1OUT
			bic.b	#0004h,		&P2IFG
			jmp		Main

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
            
