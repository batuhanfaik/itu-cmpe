; R10 = A, R11 = B, R12 = C, R13 = D, R4 = addr_primes, R5 = prime_index, R6 = prime_test, R7 = iteration, R8 = temp_modulus

			mov		#primes, R4
			mov		#primes, R5
			mov		#2, R6
			mov		#2, R7

main		cmp		#0, 98(R4)
			jne		part4
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

; R4= i, R5= j, R6= k, R7= end_prime_addr, R8= arr1_ptr, R9= arr2_ptr

part4		mov 	#200, R4
			mov		#primes, R7
			add		#98, R7
			mov		#array1, R8
			mov		#array2, R9

forI		mov		#primes, R5

forJ		mov		#primes, R6
forK		mov		0(R5), R10
			add		0(R6), R10
			cmp		R4, R10
			jz		add_comp
			cmp		R7, R6
			jz		iterJ
			add		#2, R6
			jmp		forK

iterJ		cmp		R5, R7
			jz		iterI
			add		#2, R5
			jmp 	forJ

add_comp	mov		0(R5), 0(R8)
			mov		0(R6), 0(R9)
			add		#2, R8
			add		#2, R9

iterI		cmp		#300, R4
			jz		exit
			add		#2, R4
			jmp		forI

			mov		#array1, R8
			mov		#array2, R9
exit		jmp		exit

			.data
primes		.space	100
array1		.space	100
array2		.space	100