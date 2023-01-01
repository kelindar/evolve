// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !noasm && !gccgo && !safe
// +build !noasm,!gccgo,!safe

package ffnet

// AxpyUnitary is
//
//	for i, v := range x {
//		y[i] += alpha * v
//	}
func axpyUnitary(alpha float32, x, y []float32)
