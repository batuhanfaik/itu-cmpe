; PART 1

SetupP1		bis.b 	#00h, &P1DIR ;P1.0 output
Mainloop	xor.b	#001h, &P1OUT ; Toggle P1.0
Wait		mov.w 	#250000, R15 ; Delay to R15
L1			dec.w	#R15 ; Decrement R15
			jnz L1; Delay over?
			jmp Mainloop; Again
			


; PART 2
SetupP2		mov.b 	#0FFh, &P1DIR ;Enable lights
			mov.b	#0FFh, &P2DIR ;Enable second column of lights
Reset		mov.b	#01h, &P1OUT ;1st step
			jmp Wait ;Jump to wait?
			mov.b	#80h, &P2OUT ;1st step second column

			;	Enable if the line above does not work
;Wait		mov.w 	#250000, R15 ; Delay to R15
;L1			dec.w	#R15 ; Decrement R15
			;jnz L1; Delay over?

Mainloop	rla.b	P1OUT ;
			add.b 	#01h, &P1OUT ;add 1
			rra.b	P2OUT ; rotate arithmetic right 
Wait		mov.w 	#250000, R15 ; Delay to R15
L1			dec.w	#R15 ; Decrement R15
			jnz L1; Delay over?
			cmp.b	#FFh, P2OUT; check if p2out is FF (all lights are on)
			jnz Reset ;Jump to reset
			jmp Mainloop; Again	
			

; PART 3
SetupP3		mov.b	#33h, 	&P1DIR ;Enabling only the leds that'll light up
			mov.b 	#0CCh, 	&P2DIR ; //  	// 			//				//
Reset		mov.b	#01h, 	&R14 ;	Set R14 to 000000001;
Mainloop	mov.b 	R14,	&P1OUT	; Light up the row 
			mov.b	R14, 	&P2OUT	; Only enabled ones will light up
			rla.b	R14				; Shift left so next row can light up accordinly
Wait		mov.w 	#250000, R15 ; Delay to R15
L1			dec.w	#R15 ; Decrement R15
			jnz L1; Delay over?
			cmp.b	#080h, R14; check if p2out is FF (all lights are on)
			jnz Reset ;Jump to reset
			jmp Mainloop; Again	
			
