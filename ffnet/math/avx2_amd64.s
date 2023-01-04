	.text
	.intel_syntax noprefix
	.file	"avx2_amd64.cpp"
	.globl	f32_axpy                        # -- Begin function f32_axpy
	.p2align	4, 0x90
	.type	f32_axpy,@function
f32_axpy:                               # @f32_axpy
# %bb.0:
	push	rbp
	mov	rbp, rsp
	and	rsp, -8
	cmp	rdx, 8
	jb	.LBB0_5
# %bb.1:
	vbroadcastss	ymm1, xmm0
	lea	rax, [rdx - 8]
	mov	r8, rax
	shr	r8, 3
	inc	r8
	cmp	rax, 8
	jae	.LBB0_12
# %bb.2:
	xor	ecx, ecx
	jmp	.LBB0_3
.LBB0_12:
	mov	rax, r8
	and	rax, -2
	xor	ecx, ecx
	.p2align	4, 0x90
.LBB0_13:                               # =>This Inner Loop Header: Depth=1
	vmovups	ymm2, ymmword ptr [rdi + 4*rcx]
	vfmadd213ps	ymm2, ymm1, ymmword ptr [rsi + 4*rcx] # ymm2 = (ymm1 * ymm2) + mem
	vmovups	ymmword ptr [rsi + 4*rcx], ymm2
	vmovups	ymm2, ymmword ptr [rdi + 4*rcx + 32]
	vfmadd213ps	ymm2, ymm1, ymmword ptr [rsi + 4*rcx + 32] # ymm2 = (ymm1 * ymm2) + mem
	vmovups	ymmword ptr [rsi + 4*rcx + 32], ymm2
	add	rcx, 16
	add	rax, -2
	jne	.LBB0_13
.LBB0_3:
	test	r8b, 1
	je	.LBB0_5
# %bb.4:
	vmovups	ymm2, ymmword ptr [rdi + 4*rcx]
	vfmadd213ps	ymm1, ymm2, ymmword ptr [rsi + 4*rcx] # ymm1 = (ymm2 * ymm1) + mem
	vmovups	ymmword ptr [rsi + 4*rcx], ymm1
.LBB0_5:
	test	dl, 7
	je	.LBB0_11
# %bb.6:
	mov	rax, rdx
	and	rax, -8
	cmp	rax, rdx
	jae	.LBB0_11
# %bb.7:
	mov	rcx, rax
	not	rcx
	test	dl, 1
	je	.LBB0_9
# %bb.8:
	vmovss	xmm1, dword ptr [rdi + 4*rax]   # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	xmm1, xmm0, dword ptr [rsi + 4*rax] # xmm1 = (xmm0 * xmm1) + mem
	vmovss	dword ptr [rsi + 4*rax], xmm1
	or	rax, 1
.LBB0_9:
	add	rcx, rdx
	je	.LBB0_11
	.p2align	4, 0x90
.LBB0_10:                               # =>This Inner Loop Header: Depth=1
	vmovss	xmm1, dword ptr [rdi + 4*rax]   # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	xmm1, xmm0, dword ptr [rsi + 4*rax] # xmm1 = (xmm0 * xmm1) + mem
	vmovss	dword ptr [rsi + 4*rax], xmm1
	vmovss	xmm1, dword ptr [rdi + 4*rax + 4] # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	xmm1, xmm0, dword ptr [rsi + 4*rax + 4] # xmm1 = (xmm0 * xmm1) + mem
	vmovss	dword ptr [rsi + 4*rax + 4], xmm1
	add	rax, 2
	cmp	rax, rdx
	jb	.LBB0_10
.LBB0_11:
	mov	rsp, rbp
	pop	rbp
	vzeroupper
	ret
.Lfunc_end0:
	.size	f32_axpy, .Lfunc_end0-f32_axpy
                                        # -- End function
	.globl	f32_matmul                      # -- Begin function f32_matmul
	.p2align	4, 0x90
	.type	f32_matmul,@function
f32_matmul:                             # @f32_matmul
# %bb.0:
	push	rbp
	mov	rbp, rsp
	push	r15
	push	r14
	push	r13
	push	r12
	push	rbx
	and	rsp, -8
	sub	rsp, 120
	mov	qword ptr [rsp + 16], rdx       # 8-byte Spill
	mov	qword ptr [rsp + 56], rsi       # 8-byte Spill
	mov	qword ptr [rsp + 8], rdi        # 8-byte Spill
	mov	qword ptr [rsp + 40], rcx       # 8-byte Spill
	test	rcx, rcx
	je	.LBB1_37
# %bb.1:
	mov	rax, qword ptr [rbp + 16]
	test	al, 7
	setne	cl
	mov	r9, rax
	and	r9, -8
	cmp	r9, rax
	setb	dl
	test	r8, r8
	je	.LBB1_37
# %bb.2:
	and	cl, dl
	cmp	rax, 8
	jb	.LBB1_28
# %bb.3:
	movabs	r10, 4611686018427387902
	test	cl, cl
	je	.LBB1_4
# %bb.8:
	lea	rcx, [rax - 8]
	mov	qword ptr [rsp + 32], rcx       # 8-byte Spill
	shr	rcx, 3
	inc	rcx
	mov	rdx, r9
	not	rdx
	mov	qword ptr [rsp + 48], rdx       # 8-byte Spill
	mov	rdx, rcx
	mov	qword ptr [rsp + 24], rcx       # 8-byte Spill
	and	r10, rcx
	mov	qword ptr [rsp + 104], r10      # 8-byte Spill
	mov	rcx, r9
	or	rcx, 1
	mov	qword ptr [rsp + 96], rcx       # 8-byte Spill
	mov	rcx, rax
	neg	rcx
	mov	qword ptr [rsp + 112], rcx      # 8-byte Spill
	mov	rdx, qword ptr [rsp + 8]        # 8-byte Reload
	lea	rbx, [rdx + 32]
	lea	r15, [4*rax]
	mov	rcx, qword ptr [rsp + 16]       # 8-byte Reload
	lea	rsi, [rcx + 32]
	mov	qword ptr [rsp + 80], rsi       # 8-byte Spill
	add	rcx, 4
	mov	qword ptr [rsp + 72], rcx       # 8-byte Spill
	xor	edi, edi
	mov	rcx, rdx
	mov	qword ptr [rsp + 64], r8        # 8-byte Spill
	jmp	.LBB1_9
	.p2align	4, 0x90
.LBB1_21:                               #   in Loop: Header=BB1_9 Depth=1
	mov	rdi, qword ptr [rsp + 88]       # 8-byte Reload
	inc	rdi
	add	rbx, r15
	add	rcx, r15
	cmp	rdi, qword ptr [rsp + 40]       # 8-byte Folded Reload
	mov	r8, qword ptr [rsp + 64]        # 8-byte Reload
	je	.LBB1_37
.LBB1_9:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_10 Depth 2
                                        #       Child Loop BB1_13 Depth 3
                                        #       Child Loop BB1_19 Depth 3
	mov	rdx, rdi
	imul	rdx, rax
	mov	rsi, qword ptr [rsp + 8]        # 8-byte Reload
	lea	r10, [rsi + 4*rdx]
	mov	rdx, r8
	mov	qword ptr [rsp + 88], rdi       # 8-byte Spill
	mov	r8, rdi
	imul	r8, rdx
	mov	r11, qword ptr [rsp + 72]       # 8-byte Reload
	mov	rdx, qword ptr [rsp + 80]       # 8-byte Reload
	xor	r14d, r14d
	jmp	.LBB1_10
	.p2align	4, 0x90
.LBB1_20:                               #   in Loop: Header=BB1_10 Depth=2
	inc	r14
	add	rdx, r15
	add	r11, r15
	cmp	r14, qword ptr [rsp + 64]       # 8-byte Folded Reload
	je	.LBB1_21
.LBB1_10:                               #   Parent Loop BB1_9 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_13 Depth 3
                                        #       Child Loop BB1_19 Depth 3
	mov	r12, r14
	imul	r12, rax
	lea	rsi, [r14 + r8]
	mov	rdi, qword ptr [rsp + 56]       # 8-byte Reload
	vbroadcastss	ymm0, dword ptr [rdi + 4*rsi]
	cmp	qword ptr [rsp + 32], 8         # 8-byte Folded Reload
	jae	.LBB1_12
# %bb.11:                               #   in Loop: Header=BB1_10 Depth=2
	xor	r13d, r13d
	jmp	.LBB1_14
	.p2align	4, 0x90
.LBB1_12:                               #   in Loop: Header=BB1_10 Depth=2
	mov	rdi, qword ptr [rsp + 104]      # 8-byte Reload
	xor	r13d, r13d
	.p2align	4, 0x90
.LBB1_13:                               #   Parent Loop BB1_9 Depth=1
                                        #     Parent Loop BB1_10 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovups	ymm1, ymmword ptr [rdx + 4*r13 - 32]
	vfmadd213ps	ymm1, ymm0, ymmword ptr [rbx + 4*r13 - 32] # ymm1 = (ymm0 * ymm1) + mem
	vmovups	ymmword ptr [rbx + 4*r13 - 32], ymm1
	vmovups	ymm1, ymmword ptr [rdx + 4*r13]
	vfmadd213ps	ymm1, ymm0, ymmword ptr [rbx + 4*r13] # ymm1 = (ymm0 * ymm1) + mem
	vmovups	ymmword ptr [rbx + 4*r13], ymm1
	add	r13, 16
	add	rdi, -2
	jne	.LBB1_13
.LBB1_14:                               #   in Loop: Header=BB1_10 Depth=2
	mov	rsi, qword ptr [rsp + 16]       # 8-byte Reload
	lea	rsi, [rsi + 4*r12]
	test	byte ptr [rsp + 24], 1          # 1-byte Folded Reload
	jne	.LBB1_15
# %bb.16:                               #   in Loop: Header=BB1_10 Depth=2
	mov	rdi, r9
	test	al, 1
	jne	.LBB1_17
.LBB1_18:                               #   in Loop: Header=BB1_10 Depth=2
	mov	rsi, qword ptr [rsp + 112]      # 8-byte Reload
	cmp	qword ptr [rsp + 48], rsi       # 8-byte Folded Reload
	jne	.LBB1_19
	jmp	.LBB1_20
	.p2align	4, 0x90
.LBB1_15:                               #   in Loop: Header=BB1_10 Depth=2
	vmovups	ymm1, ymmword ptr [rsi + 4*r13]
	vfmadd213ps	ymm1, ymm0, ymmword ptr [r10 + 4*r13] # ymm1 = (ymm0 * ymm1) + mem
	vmovups	ymmword ptr [r10 + 4*r13], ymm1
	mov	rdi, r9
	test	al, 1
	je	.LBB1_18
.LBB1_17:                               #   in Loop: Header=BB1_10 Depth=2
	vmovss	xmm1, dword ptr [rsi + 4*r9]    # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	xmm1, xmm0, dword ptr [r10 + 4*r9] # xmm1 = (xmm0 * xmm1) + mem
	vmovss	dword ptr [r10 + 4*r9], xmm1
	mov	rdi, qword ptr [rsp + 96]       # 8-byte Reload
	mov	rsi, qword ptr [rsp + 112]      # 8-byte Reload
	cmp	qword ptr [rsp + 48], rsi       # 8-byte Folded Reload
	je	.LBB1_20
	.p2align	4, 0x90
.LBB1_19:                               #   Parent Loop BB1_9 Depth=1
                                        #     Parent Loop BB1_10 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovss	xmm1, dword ptr [r11 + 4*rdi - 4] # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	xmm1, xmm0, dword ptr [rcx + 4*rdi] # xmm1 = (xmm0 * xmm1) + mem
	vmovss	dword ptr [rcx + 4*rdi], xmm1
	vmovss	xmm1, dword ptr [r11 + 4*rdi]   # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	xmm1, xmm0, dword ptr [rcx + 4*rdi + 4] # xmm1 = (xmm0 * xmm1) + mem
	vmovss	dword ptr [rcx + 4*rdi + 4], xmm1
	add	rdi, 2
	cmp	rdi, rax
	jb	.LBB1_19
	jmp	.LBB1_20
.LBB1_28:
	test	cl, cl
	je	.LBB1_37
# %bb.29:
	mov	r15, r9
	not	r15
	mov	rcx, r9
	or	rcx, 1
	mov	qword ptr [rsp + 32], rcx       # 8-byte Spill
	mov	r12, rax
	neg	r12
	lea	r13, [4*rax]
	mov	rcx, qword ptr [rsp + 16]       # 8-byte Reload
	add	rcx, 4
	mov	qword ptr [rsp + 48], rcx       # 8-byte Spill
	xor	edi, edi
	mov	rdx, qword ptr [rsp + 8]        # 8-byte Reload
	mov	r11, qword ptr [rsp + 16]       # 8-byte Reload
	jmp	.LBB1_30
	.p2align	4, 0x90
.LBB1_36:                               #   in Loop: Header=BB1_30 Depth=1
	mov	rdi, qword ptr [rsp + 24]       # 8-byte Reload
	inc	rdi
	add	rdx, r13
	cmp	rdi, qword ptr [rsp + 40]       # 8-byte Folded Reload
	je	.LBB1_37
.LBB1_30:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_31 Depth 2
                                        #       Child Loop BB1_34 Depth 3
	mov	rcx, rdi
	imul	rcx, rax
	mov	rsi, qword ptr [rsp + 8]        # 8-byte Reload
	lea	r14, [rsi + 4*rcx]
	mov	qword ptr [rsp + 24], rdi       # 8-byte Spill
	imul	rdi, r8
	mov	rbx, qword ptr [rsp + 48]       # 8-byte Reload
	xor	r10d, r10d
	jmp	.LBB1_31
	.p2align	4, 0x90
.LBB1_35:                               #   in Loop: Header=BB1_31 Depth=2
	inc	r10
	add	rbx, r13
	cmp	r10, r8
	je	.LBB1_36
.LBB1_31:                               #   Parent Loop BB1_30 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_34 Depth 3
	lea	rsi, [r10 + rdi]
	mov	rcx, qword ptr [rsp + 56]       # 8-byte Reload
	vmovss	xmm0, dword ptr [rcx + 4*rsi]   # xmm0 = mem[0],zero,zero,zero
	mov	rsi, r9
	test	al, 1
	je	.LBB1_33
# %bb.32:                               #   in Loop: Header=BB1_31 Depth=2
	mov	rsi, r10
	imul	rsi, rax
	lea	rsi, [r11 + 4*rsi]
	vmovss	xmm1, dword ptr [rsi + 4*r9]    # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	xmm1, xmm0, dword ptr [r14 + 4*r9] # xmm1 = (xmm0 * xmm1) + mem
	vmovss	dword ptr [r14 + 4*r9], xmm1
	mov	rsi, qword ptr [rsp + 32]       # 8-byte Reload
.LBB1_33:                               #   in Loop: Header=BB1_31 Depth=2
	cmp	r15, r12
	je	.LBB1_35
	.p2align	4, 0x90
.LBB1_34:                               #   Parent Loop BB1_30 Depth=1
                                        #     Parent Loop BB1_31 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovss	xmm1, dword ptr [rbx + 4*rsi - 4] # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	xmm1, xmm0, dword ptr [rdx + 4*rsi] # xmm1 = (xmm0 * xmm1) + mem
	vmovss	dword ptr [rdx + 4*rsi], xmm1
	vmovss	xmm1, dword ptr [rbx + 4*rsi]   # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	xmm1, xmm0, dword ptr [rdx + 4*rsi + 4] # xmm1 = (xmm0 * xmm1) + mem
	vmovss	dword ptr [rdx + 4*rsi + 4], xmm1
	add	rsi, 2
	cmp	rsi, rax
	jb	.LBB1_34
	jmp	.LBB1_35
.LBB1_4:
	lea	r11, [rax - 8]
	mov	r14, r11
	shr	r14, 3
	inc	r14
	and	r10, r14
	mov	rcx, qword ptr [rsp + 8]        # 8-byte Reload
	lea	rdi, [rcx + 32]
	lea	r15, [4*rax]
	mov	rcx, qword ptr [rsp + 16]       # 8-byte Reload
	add	rcx, 32
	mov	qword ptr [rsp + 24], rcx       # 8-byte Spill
	xor	r13d, r13d
	mov	r9, qword ptr [rsp + 16]        # 8-byte Reload
	jmp	.LBB1_5
	.p2align	4, 0x90
.LBB1_27:                               #   in Loop: Header=BB1_5 Depth=1
	mov	r13, qword ptr [rsp + 32]       # 8-byte Reload
	inc	r13
	add	rdi, r15
	cmp	r13, qword ptr [rsp + 40]       # 8-byte Folded Reload
	je	.LBB1_37
.LBB1_5:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_6 Depth 2
                                        #       Child Loop BB1_23 Depth 3
	mov	rcx, r13
	imul	rcx, rax
	mov	rdx, qword ptr [rsp + 8]        # 8-byte Reload
	lea	r12, [rdx + 4*rcx]
	mov	qword ptr [rsp + 32], r13       # 8-byte Spill
	imul	r13, r8
	mov	rbx, qword ptr [rsp + 24]       # 8-byte Reload
	xor	ecx, ecx
	jmp	.LBB1_6
	.p2align	4, 0x90
.LBB1_26:                               #   in Loop: Header=BB1_6 Depth=2
	inc	rcx
	add	rbx, r15
	cmp	rcx, r8
	je	.LBB1_27
.LBB1_6:                                #   Parent Loop BB1_5 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB1_23 Depth 3
	lea	rdx, [rcx + r13]
	mov	rsi, qword ptr [rsp + 56]       # 8-byte Reload
	vbroadcastss	ymm0, dword ptr [rsi + 4*rdx]
	cmp	r11, 8
	jae	.LBB1_22
# %bb.7:                                #   in Loop: Header=BB1_6 Depth=2
	xor	edx, edx
	jmp	.LBB1_24
	.p2align	4, 0x90
.LBB1_22:                               #   in Loop: Header=BB1_6 Depth=2
	mov	rsi, r10
	xor	edx, edx
	.p2align	4, 0x90
.LBB1_23:                               #   Parent Loop BB1_5 Depth=1
                                        #     Parent Loop BB1_6 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovups	ymm1, ymmword ptr [rbx + 4*rdx - 32]
	vfmadd213ps	ymm1, ymm0, ymmword ptr [rdi + 4*rdx - 32] # ymm1 = (ymm0 * ymm1) + mem
	vmovups	ymmword ptr [rdi + 4*rdx - 32], ymm1
	vmovups	ymm1, ymmword ptr [rbx + 4*rdx]
	vfmadd213ps	ymm1, ymm0, ymmword ptr [rdi + 4*rdx] # ymm1 = (ymm0 * ymm1) + mem
	vmovups	ymmword ptr [rdi + 4*rdx], ymm1
	add	rdx, 16
	add	rsi, -2
	jne	.LBB1_23
.LBB1_24:                               #   in Loop: Header=BB1_6 Depth=2
	test	r14b, 1
	je	.LBB1_26
# %bb.25:                               #   in Loop: Header=BB1_6 Depth=2
	mov	rsi, rcx
	imul	rsi, rax
	lea	rsi, [r9 + 4*rsi]
	vmovups	ymm1, ymmword ptr [rsi + 4*rdx]
	vfmadd213ps	ymm0, ymm1, ymmword ptr [r12 + 4*rdx] # ymm0 = (ymm1 * ymm0) + mem
	vmovups	ymmword ptr [r12 + 4*rdx], ymm0
	jmp	.LBB1_26
.LBB1_37:
	lea	rsp, [rbp - 40]
	pop	rbx
	pop	r12
	pop	r13
	pop	r14
	pop	r15
	pop	rbp
	vzeroupper
	ret
.Lfunc_end1:
	.size	f32_matmul, .Lfunc_end1-f32_matmul
                                        # -- End function
	.ident	"Ubuntu clang version 15.0.6"
	.section	".note.GNU-stack","",@progbits
	.addrsig
