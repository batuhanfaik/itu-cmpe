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
			mov.b	#00000100b,	&P2OUT
ButtonLoop	bit.b	#00100000b,	&P1IN
			jz		ButtonLoop
			xor.b	#001100b,	&P2OUT
PressLoop	bit.b	#00100000b,	&P1IN
			jnz		PressLoop
Wait		mov.w	#250000,	R15
L1			dec.w	R15
			jnz		L1
			jmp		ButtonLoop

;			Part 3
SetupLED	mov.b	#00FFh,		&P1DIR
			bic.b	#0002h,		&P2DIR
			mov.b	#00000000b,	&P1OUT
ButtonLoop	bit.b	#00000010b,	&P2IN
			jz		ButtonLoop
			inc.b	Counter
			mov.b	Counter,	&P1OUT
PressLoop	bit.b	#00000010b,	&P2IN
			jnz		PressLoop
Wait		mov.w	#250000,	R15
L1			dec.w	R15
			jnz		L1
			jmp		ButtonLoop

			.data
Counter		.word	0

;			Part 4
SetupLED	mov.b	#00FFh,		&P1DIR
			bic.b	#00001110b,	&P2DIR
			mov.b	#00000000b,	&P1OUT
ButtonLoop	bit.b	#00000010b,	&P2IN
			jnz		Increment
			bit.b	#00000100b,	&P2IN
			jnz		Complement
			bit.b	#00001000b,	&P2IN
			jnz		Reset
			jz		ButtonLoop
Reset		mov.b	#0000h,		&P1OUT
			mov.b	#0000h,		Counter
			jmp		PressLoop
Complement	xor.b	#0FFh,		Counter
			mov.b	Counter,	&P1OUT
			jmp		PressLoop
Increment	inc.b	Counter
			mov.b	Counter,	&P1OUT
PressLoop	bit.b	#00000100b,	&P2IN
			jnz		PressLoop
			bit.b	#00000010b,	&P2IN
			jnz		PressLoop
			bit.b	#00001000b,	&P2IN
			jnz		PressLoop
Wait		mov.w	#250000,	R15
L1			dec.w	R15
			jnz		L1
			jmp		ButtonLoop

			.data
Counter		.word	0

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
