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
; R10 = A, R11 = B, R12 = C, R13 = D, R4 = addr_primes, R5 = prime_index, R6 = prime_test, R7 = iteration, R8 = temp_modulus

			mov		#primes, R4
			mov		#primes, R5
			mov		#2, R6
			mov		#2, R7

main		cmp		#0, 98(R4)
			jne		exit
			mov		R6, R10
			mov		R7, R11
			jmp		modulus

append		cmp		#0 ,R13
			jne		addition
			cmp		R6, R7
			jeq		add_prime
			inc		R6
			mov		#2, R7
			jmp		main

addition	inc		R7
			mov		R7,	R11
			jmp		modulus

add_prime   mov		R6, 0(R5)
			add		#2, R5
			inc		R6
			mov		#2, R7
			jmp		main

modulus		mov		R11, R12
			mov		R10, R13
			mov 	R10, R8

			rra		R8
cLEthan		cmp		R12, R8
			jge		multiply
			jmp		bLEthan

multiply	rla		R12
			jmp		cLEthan

bLEthan		cmp 	R11, R13
			jge		subtract
			jmp		append

subtract	cmp		R12, R13
			jl		divide
			sub		R12, R13
			jmp		divide

divide		rra		R12
			jmp		bLEthan

exit		jmp		exit

			.data
primes		.space	100

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
            
