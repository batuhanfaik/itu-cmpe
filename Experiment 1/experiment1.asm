; PART 1

SetupP1		bis.b 	#00h, &P1DIR ;P1.0 output
Mainloop	xor.b	#001h, &P1OUT ; Toggle P1.0
Wait		mov.w 	#050000, R15 ; Delay to R15
L1			dec.w	#R15 ; Decrement R15
			jnz L1; Delay over?
			jmp Mainloop; Again
			


; PART 2
SetupP2		bis.b 	#FFh, #P1DIR ;Enable lights
Mainloop