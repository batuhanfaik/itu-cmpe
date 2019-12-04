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
			jz		endseq
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

endseq		jmp stop


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



			.data
string 		.byte "ITU - Comp. Eng.",0Dh,"MC Lab. 2019",00h

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
            
