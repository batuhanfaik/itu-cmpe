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
; R10 = A, R11 = B, R12 = C, R13 = D
initial		mov		#151, R10
			mov		#8, R11

main		mov		R11, R12
			mov		R10, R13
			mov 	R10, R9

			rra		R4
cLEthan		cmp		R12, R9
			jge		multiply
			jmp		bLEthan

multiply	rla		R12
			jmp		cLEthan

bLEthan		cmp 	R11, R13
			jge		subtract
			jmp		exit

subtract	cmp		R12, R13
			jl		divide
			sub		R12, R13
			jmp		divide

divide		rra		R12
			jmp		bLEthan

exit		jmp		exit

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
            
