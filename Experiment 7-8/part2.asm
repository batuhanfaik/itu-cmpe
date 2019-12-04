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

setup_INT   bis.b   #020h,      &P2IE       ;Enabling interrupt
            bis.b   #020h,      &P2IES      ; High to low interrupt mode, only works when you let go of the button
            clr     &P2IFG ; Clearing flags
            eint    ;Enabling interrupt


Setup	 	mov.b #0ffh, &P1DIR
			mov.b #0ffh, &P2DIR
			mov.b #00000000b, &P2SEL
			mov.b #00000000b, &P2SEL2
			clr.b &P1OUT
			clr.b &P2OUT

InitLCD		mov &Delay100ms, R15
			call #Delay

			mov.b #00110000b, &P1OUT
			call #TrigEn
			mov &Delay4ms, R15
			call #Delay

			call #TrigEn
			mov &Delay100us, R15
			call #Delay

			call #TrigEn
			mov &Delay100us, R15
			call #Delay

			mov.b #00100000b, &P1OUT
			call #TrigEn
			mov &Delay100us, R15
			call #Delay

			mov.b #00100000b, &P1OUT
			call #TrigEn
			mov.b #10000000b, &P1OUT
			call #TrigEn

			mov &Delay100us, R15
			call #Delay

			mov.b #00000000b, &P1OUT
			call #TrigEn
			mov.b #10000000b, &P1OUT
			call #TrigEn

			mov &Delay100us, R15
			call #Delay

			mov.b #00000000b, &P1OUT
			call #TrigEn
			mov.b #00010000b, &P1OUT
			call #TrigEn

			mov &Delay4ms, R15
			call #Delay

			mov.b #00000000b, &P1OUT
			call #TrigEn
			mov.b #01100000b, &P1OUT
			call #TrigEn
			mov &Delay100us, R15
			call #Delay

Main		clr R5
			mov.b #00001111b, R5
			call #SendCMD
			mov &Delay50us, R15
			call #Delay

;initialization is over. Now we will start sending data to our LCD display

;------------------

;Write your code


;r10= address pointer
			mov.w #string,	R10

loop		cmp.b #0dh,		0(R10)
			jz		newline
			cmp.b	#00h,	0(R10)
			jz		writenumber
			mov.b	@R10, R5
			call 	#SendData
			mov     &Delay100us, R15
			call 	#Delay
			add.b	#01b,	R10
			jmp loop

newline		add.b	#01b,	R10
			mov.b	#00100000b, R15	;0,0,1,I,n,f,x,x fix this part
			call 	#SendCMD
			mov 	&Delay100us, R15
			call 	#Delay
			mov.b	#00000010b, R15	; send cursor back to home
			call 	#SendCMD
			mov 	&Delay100us, R15
			call 	#Delay
			jmp		loop

writenumber
;todo



SendData 	bis.b #080h,  &P2OUT
			mov.b R5, &P1OUT
			call #TrigEn
			rla R5
			rla R5
			rla R5
			rla R5
			mov.b R5, &P1OUT
			call #TrigEn
			bic.b 	#080h,	&P2OUT
			ret


stop 		mov.b	&Delay100us,	R15
			call 	#Delay
			jmp 	stop

;------------------
;	I 			S			R
;----------------------
ISR         dint
;todo
            clr     &P2IFG
            eint
            reti
;----------------------

TrigEn      bis.b #01000000b, &P2OUT
			bic.b #01000000b, &P2OUT
			ret

SendCMD     mov.b R5, &P1OUT

			call #TrigEn
			rla R5
			rla R5
			rla R5
			rla R5
			mov.b R5, &P1OUT
			call #TrigEn
			ret

Delay       dec.w R15 ; Decrement R15
			jnz Delay
			ret
;--------------------------------
Mul_func	push	r4
			push	r5
			push	r6
			mov		#0, r6
			mov		8(sp), r4
			mov		10(sp), r5
for_mul		cmp		#0, r5
			jz		mul_return
			add		r4, r6
			dec		r5
			jmp		for_mul
mul_return	mov		r6, 12(sp)
			pop		r6
			pop		r5
			pop		r4
			ret

Div_func	push	r4
			push	r5
			push	r6
			mov		#0, r6
			mov		8(sp), r4
			mov		10(sp), r5
for_div		cmp		r5, r4
			jl		div_return
			sub		r5, r4
			inc		r6
			jmp		for_div
div_return	mov		r6, 12(sp)
			pop		r6
			pop		r5
			pop		r4
			ret
;--------------------------------

			.data
string 		.byte "ITU - Comp. Eng.",0Dh,"Number: ",00h

Delay50us   .word   011h
Delay100us  .word   022h
Delay2ms    .word   0250h
Delay4ms    .word   0510h
Delay100ms  .word   07A10h
                                            

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
            .sect 	".int03"
            .short	ISR
