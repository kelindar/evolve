// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !amd64 || noasm || gccgo || safe
// +build !amd64 noasm gccgo safe

package ffnet

// axpyUnitary is
//
//	for i, v := range x {
//		y[i] += alpha * v
//	}
func axpyUnitary(alpha float32, x, y []float32) {
	for i, v := range x {
		y[i] += alpha * v
	}
}
