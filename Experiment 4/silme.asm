; R4 = a, R5 = b, R6 = result

__main		push	0
			push    #5
			push	#2
			call	Add_func
return		pop		r4
			pop		r5
			pop		r6

Add_func	push	r4
			push	r5
			mov		6(sp), r4
			mov		8(sp), r5
			add		r5, r4
			mov		r4, 10(sp)
			pop		r5
			pop		r4
			ret

Sub_func	push	r4
			push	r5
			mov		6(sp), r4
			mov		8(sp), r5
			sub		r5, r4
			mov		r4, 10(sp)
			pop		r5
			pop		r4
			ret

; R6 = i
Mul_func	push	r4
			push	r5
			push	r6
			mov		#0, r6
			mov		8(sp), r4
			mov		10(sp), r5
for_mul		cmp		r5, #0
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

Perm_func	push	r4
			push	r5
			push	r6
			mov		#1, r6
			mov		8(sp), r4
			mov		10(sp), r5
for_perm	cmp		#0, r5
			jz		perm_return

			push	r6
			push	r6
			push	r4
			call	Mul_func

			pop		r4
			pop		r6
			pop		r6

			dec		r4
			dec		r5
			jmp		for_perm
perm_return	mov		r6, 12(sp)
			pop		r6
			pop		r5
			pop		r4
			ret

Fact_func	push	r4
			push	r5
			mov		6(sp), r4
			cmp		#2, r4
			jl		fact_rtrn
			mov		r4, r5
			dec		r5
			push	r5
			push	r5
			call	Fact_func
			pop		r5
			pop		r5
			push	r4
			push	r5
			push	r4
			call	Mul_func
			pop		r4
			pop		r4
			pop		r4
			mov		r4, 4(sp)
			pop		r5
			pop		r4
			ret
fact_rtrn   mov		#1, 4(sp)
			pop		r5
			pop		r4
			ret
			
			result		.bss		resultArray, 5		; Memory allocate
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

Setup		mov	#array, r5
			mov #resultArray, r10

Mainloop	mov.b	@r5, r6
			inc		r5
			call	#func1
			mov.b	r6, 0(r10)
			inc		r10
			cmp		#lastElement, r5
			jlo		Mainloop
			jmp		finish

func1		xor.b	#0FFh, r6
			mov.b	r6, r7
			call	#func2
			mov.b	r7, r6
			ret

func2		inc.b	r7
			ret

array		.byte 127, -128,0,55
lastElement

finish		nop