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

setup_INT   bis.b   #040h,      &P2IE       ;Enabling interrupt
            and.b   #0BFh,      &P2SEL      ; 0 = general purpose I/O function
            and.b   #0BFh,      &P2SEL2     ; 1 = peripheral module function
            bis.b   #040h,      &P2IES      ; High to low interrupt mode, only works when you let go of the button
            clr     &P2IFG ; Clearing flags
            eint    ;Enabling interrupt


Setup		bis.b	#0FFh,		&P1DIR
			mov.b	#00Fh,		&P2DIR
			bic.b	#0FFh,		&P1OUT
			mov.b	#001h,		&P2OUT
			mov.w	#arr, 		r4
            mov.w   #ach_arr,   r5
            mov.w   #0000h,     r6
; r4= count_pointer, r5=achilles_pointer, r6=dipslay status
; r6= 0, type achilles
; r6= 1, show count 


Main        cmp     #001h,      r6
            jz      Show_count
            mov.b   @r5,        &P1OUT ; Typing achilles character by character to screen
            call    Delay
            add     #001h,      r5
            cmp     #ach_end,   r5
            jz      Reset_seq
            jmp     Main

; Display how many times achilles has been written on the screen
Show_count  mov.b   @r4,        &P1OUT
            call    Delay
            jmp     Main

; Reset count
Reset_count	mov.b	#arr,		r4
			jmp     Main
; Reset character sequence
Reset_seq   mov.w   #ach_arr,   r5
            add     #001h,      r4  ; Increment the counter since we typed the word  
            cmp     #arr_end    r4  ; Reset if count goes over 9
            jz      Reset_count
            jmp     Main

; Delay function
Delay		push 	r14
			push    r15
			mov.w	#0Ah,		R14
L2			mov.w	#07A00h,	R15
L1			dec.w	R15
			jnz		L1
			dec.w	R14
			jnz		L2
			pop		r15
			pop 	r14
			ret

ISR         dint
            xor.b   #001h,      r6 ;Changing mode
            clr     &P2IFG
            eint
            reti


;						0			1		   2	     3			4			5		  6			  7			8		   9
arr			.byte	00111111b, 00000110b, 01011011b, 01001111b, 01100110b, 01101101b, 01111101b, 00000111b, 01111111b, 01101111b
arr_end
;                       A          C          H         I           L          L           E         S
ach_arr  .byte   01110111b, 00111001b, 01110110b, 00000110b, 00111000b, 00111000b, 01111001b, 01101101b
ach_end

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
            .sect   ".int03"
            .short  ISR

            
