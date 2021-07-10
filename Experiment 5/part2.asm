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

Setup		bis.b	#0FFh,		&P1DIR
			bis.b	#00Fh,		&P2DIR
			bic.b	#0FFh,		&P1OUT
			mov.b	#001h,		&P2OUT
			mov.w	#arr, 		r4

Main		cmp		#arr_end, 	r4
			jz		Reset_seq
			mov.b	@r4,		&P1OUT
			call	#Delay
			add		#001h,		r4
			jmp		Main


; Reset sequence
Reset_seq	mov.w	#arr,		r4
			jmp     Main

; Delay function
Delay		push 	r14
			push    r15
			mov.w	#0Ah,		R14
L2			mov.w	#07A00h,	R15
L1			dec.w	R15
			jnz		L1
			dec.w	R14
			jnz		L2
			pop		r15
			pop 	r14
			ret

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
            
