// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

//go:build !amd64
// +build !amd64

package math32

import "unsafe"

//go:noescape,nosplit
func _f32_axpy(x, y unsafe.Pointer, size uint64, alpha float32) {
	panic("not supported")
}

//go:noescape,nosplit
func _f32_matmul(dst, m, n unsafe.Pointer, mr, mc, nr, nc uint64) {
	panic("not supported")
}
