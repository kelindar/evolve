// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

// Crossover function which produces a offspring from 2 parents
type Crossover = func(p1, p2, dst Evolver)

// CrossoverBestFit implements a best fit crossover
func CrossoverBestFit(p1, p2, dst Evolver) {
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
	v1, v2, out := p1.Chromosome(), p2.Chromosome(), dst.Chromosome()
	for i := range v1 {
		out[i] = (v1[i] + v2[i]) / 2
	}
}

// CrossoverMidpoint1 implements a midpoint crossover
func CrossoverMidpoint1(p1, p2, dst Evolver) {
	v1, v2, out := p1.Chromosome(), p2.Chromosome(), dst.Chromosome()
	for i := range v1 {
		out[i] = (v1[i] & 0xf0) ^ (v2[i] & 0x0f)
	}
}

// CrossoverMidpoint2 implements a midpoint crossover
func CrossoverMidpoint2(p1, p2, dst Evolver) {
	v1, v2, out := p1.Chromosome(), p2.Chromosome(), dst.Chromosome()
	for i := range v1 {
		out[i] = (v1[i] & 0x0f) ^ (v2[i] & 0xf0)
	}
}

// CrossoverMax implements a max crossover
func CrossoverMax(p1, p2, dst Evolver) {
	v1, v2, out := p1.Chromosome(), p2.Chromosome(), dst.Chromosome()
	for i := range v1 {
		out[i] = v1[i]
		if v2[i] > v1[i] {
			out[i] = v2[i]
		}
	}
}

// CrossoverMin implements a max crossover
func CrossoverMin(p1, p2, dst Evolver) {
	v1, v2, out := p1.Chromosome(), p2.Chromosome(), dst.Chromosome()
	for i := range v1 {
		out[i] = v1[i]
		if v2[i] < v1[i] {
			out[i] = v2[i]
		}
	}
}
