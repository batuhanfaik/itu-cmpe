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

;			Part 2
SetupLED	mov.b	#000Ch,		&P2DIR
			bic.b	#0020h,		&P1DIR
			mov.b	#0020h,		&P1IE
			bic.b	#0020h,		&P1IES
			mov.b	#0000h,		&P1IFG
			mov.b	#00000100b, &P2OUT
Main		bit.b	#00010000b,	&P1IFG
			jnz		Swap
			jmp 	Main
Swap		cmp.b	#00000100b,	&P2OUT
			jz		Light3
			mov.b	#00000100b,	&P2OUT
			jmp		ResetIFG
Light3		mov.b	#00000100b,	&P2OUT
			jmp		ResetIFG
ResetIFG	bic.b	#00100000b,	&P1IES
			jmp Main

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
