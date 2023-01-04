package ffnet

import "unsafe"

//go:noescape,nosplit
func _f32_axpy(x, y unsafe.Pointer, size uint64, alpha float32)

//go:noescape,nosplit
func _f32_matmul(dst, m, n unsafe.Pointer, mr, mc, nr, nc uint64)
