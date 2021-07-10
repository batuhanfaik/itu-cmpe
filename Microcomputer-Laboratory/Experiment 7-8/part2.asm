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
			mov.b	#00h,	R6 ; 100
			mov.b	#00h,	R7 ; 10
			mov.b	#00h,	R8 ; 1
			mov.b	#000d,	R9 ; random number
			mov.b	#0031d,	R11; seed
			mov.b	#00h,	R13; x
			mov.b	#00h,	R14; w


InitLCD		mov.b	#00h,	R12; rewrite on screen flag
			mov.b #0ffh, &P2DIR
			mov &Delay100ms, R15
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
			mov.w	#string,	R10

loop		cmp.b   #0dh,	0(R10)
			jz		newline
			cmp.b	#00h,	0(R10)
			jz		writenumber
			mov.b	@R10, R5
			call 	#SendData
			mov     &Delay100us, R15
			call 	#Delay
			add.w	#01b,	R10
			jmp		loop

newline		add.w	#01b,	R10
			mov.b	#011000000b, R5	;0,0,1,I,n,f,x,x fix this part
			call 	#SendCMD
			mov 	&Delay100us, R15
			call 	#Delay
			jmp		loop

; R9 = incoming generated number
; R10 = temporary valla
writenumber	; Find hundreds digit
			sub.w	#02h,	sp
			push	#100d
			push	r9
			call	#Div_func
			add.w	#04h,	sp
			pop		r6

			; Find tens digit
			sub.w	#02h,	sp
			push	#10d
			push	r9
			call	#Div_func
			add.w	#04h,	sp
			pop		r7
			sub.w	#02h,	sp
			push	#10d
			push	r6
			call	#Mul_func
			add.w	#04h,	sp
			pop		r10
			sub.b	r10,	r7

			; Find units digit
			add.b	r7,		r10
			sub.w	#02h,	sp
			push	#10d
			push	r10
			call	#Mul_func
			add.w	#04h,	sp
			pop		r10
			sub.b	r10,	r9
			mov.b	r9,		r8

			add.b	#048d, R6
			add.b	#048d, R7
			add.b	#048d, R8
			mov.b	R6, R5
			call 	#SendData
			mov     &Delay100us, R15
			call 	#Delay
			mov.b	R7, R5
			call 	#SendData
			mov     &Delay100us, R15
			call 	#Delay
			mov.b	R8, R5
			call 	#SendData
			mov     &Delay100us, R15
			call 	#Delay
			mov.b	#000h,	&P2DIR
			jmp		stop


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
			cmp.b	#01h,	r12
			jz		InitLCD
			jmp 	stop

;------------------
;	I 			S			R
;----------------------
ISR         dint
			push	r6
			sub.w	#02h,	sp
			push	r13
			push	r13
			call	#Mul_func
			add.w	#04h,	sp
			pop		r13

			add.b	r11,	r14
			add.b	r14,	r13
			mov.b	r13,	r9
			rra.b	r9
			rra.b	r9
			rra.b	r9
			rra.b	r9
			mov.b	r13,	r6
			rla.b	r6
			rla.b	r6
			rla.b	r6
			rla.b	r6
			bis.b	r6,		r9

			; Change seed
			add.b	#01h,	r11
			; I was in an interrupt flag
			mov.b	#01h,	r12

			pop		r6
            clr     &P2IFG
            eint
            reti
;----------------------

TrigEn      bis.b #01000000b, &P2OUT
			bic.b #01000000b, &P2OUT
			ret

SendCMD     mov.b R5, &P1OUT
			bic.b	#080h,	&P2OUT
			call #TrigEn
			rla R5
			rla R5
			rla R5
			rla R5
			mov.b R5, &P1OUT
			call #TrigEn
			bis.b	#080h,	&P2OUT
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
