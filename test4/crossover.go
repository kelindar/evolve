// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

import (
	"github.com/kelindar/rand"
)

// randByte generates a random byte
func randByte() byte {
	return byte(rand.Uint32n(256))
}

// CrossoverBinary implements a random binary crossover
func CrossoverBinary(p1, p2 Evolver, dst Genome) {
	v1, v2 := p1.Genome(), p2.Genome()

	n := len(v1)
	for i := 0; i < n; i++ {
		r := randByte()
		dst[i] = (v1[i] & byte(r)) ^ (v2[i] & (^byte(r)))
	}
}

func CrossoverMidpoint(p1, p2 Evolver, dst Genome) {
	v1, v2 := p1.Genome(), p2.Genome()
	for i := range v1 {
		dst[i] = (v1[i] & 0x0f) ^ (v2[i] & 0xf0)
	}
}

// CrossoverBestFit implements a best fit crossover
/*func CrossoverBestFit(p1, p2, dst Evolver) {
	funcs := []Crossover{
		CrossoverAverage,
		CrossoverMidpoint1,
		CrossoverMidpoint2,
		CrossoverMax,
		CrossoverMin,
	}

	cx, max := CrossoverAverage, float32(0)
	for _, fn := range funcs {
		fn(p1, p2, dst)
		if f := dst.Fitness(); f > max {
			cx = fn
		}
	}

	cx(p1, p2, dst)
}

// CrossoverAverage implements an average crossover
func CrossoverAverage(p1, p2, dst Evolver) {
	v1, v2, out := p1.Genome(), p2.Genome(), dst.Genome()
	for i := range v1 {
		out[i] = (v1[i] + v2[i]) / 2
	}
}

// CrossoverMidpoint1 implements a midpoint crossover
func CrossoverMidpoint1(p1, p2, dst Evolver) {
	v1, v2, out := p1.Genome(), p2.Genome(), dst.Genome()
	for i := range v1 {
		out[i] = (v1[i] & 0xf0) ^ (v2[i] & 0x0f)
	}
}

// CrossoverMidpoint2 implements a midpoint crossover
func CrossoverMidpoint2(p1, p2, dst Evolver) {
	v1, v2, out := p1.Genome(), p2.Genome(), dst.Genome()
	for i := range v1 {
		out[i] = (v1[i] & 0x0f) ^ (v2[i] & 0xf0)
	}
}

// CrossoverMax implements a max crossover
func CrossoverMax(p1, p2, dst Evolver) {
	v1, v2, out := p1.Genome(), p2.Genome(), dst.Genome()
	for i := range v1 {
		out[i] = v1[i]
		if v2[i] > v1[i] {
			out[i] = v2[i]
		}
	}
}

// CrossoverMin implements a max crossover
func CrossoverMin(p1, p2, dst Evolver) {
	v1, v2, out := p1.Genome(), p2.Genome(), dst.Genome()
	for i := range v1 {
		out[i] = v1[i]
		if v2[i] < v1[i] {
			out[i] = v2[i]
		}
	}
}
*/
