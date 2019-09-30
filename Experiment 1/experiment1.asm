; PART 1

SetupP1		bis.b 	#00h, &P1DIR ;P1.0 output
Mainloop	xor.b	#001h, &P1OUT ; Toggle P1.0
Wait		mov.w 	#250000, R15 ; Delay to R15
L1			dec.w	#R15 ; Decrement R15
			jnz L1; Delay over?
			jmp Mainloop; Again
			


; PART 2
SetupP2		mov.b 	#FFh, &P1DIR ;Enable lights
			mov.b	#FFh, &P2DIR ;Enable second column of lights
Reset		mov.b	#01h, &P1OUT ;1st step
			mov.b	#80h, &P2OUT ;1st step second column
Mainloop	rla.b	P1OUT ;
			add.b 	#01h, &P1OUT ;add 1
			rra.b	P2OUT ; rotate arithmetic right 
Wait		mov.w 	#250000, R15 ; Delay to R15
L1			dec.w	#R15 ; Decrement R15
			jnz L1; Delay over?
			cmp.b	#FFh, P2OUT; check if p2out is FF (all lights are on)
			jnz Reset ;Jump to reset
			jmp Mainloop; Again			
			
