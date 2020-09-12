// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

import (
	"math"
)

// Crossover function which produces a offspring from 2 parents
type Crossover = func(p1, p2, dst Evolver, pop []Evolver)

// CrossoverAverage implements an average crossover
func CrossoverAverage(p1, p2, dst Evolver, _ []Evolver) {
	v1, v2, out := p1.Chromosome(), p2.Chromosome(), dst.Chromosome()
	for i := range v1 {
		out[i] = (v1[i] + v2[i]) / 2
	}
}

// CrossoverAverageBound implements average-bound crossover
// ref: https://opus.lib.uts.edu.au/bitstream/10453/14522/1/2010004231.pdf
func CrossoverAverageBound(p1, p2, dst Evolver, pop []Evolver) {
	const wa, wb = 0.5, 0.5

	v1 := vector(p1.Chromosome())
	v2 := vector(p2.Chromosome())
	pMin, pMax := boundsOf(pop...)
	vMin, vMax := boundsOf(p1, p2)
	out := make([]vector, 4)

	// Compute averages and bounds
	out[0] = v1.Clone().Average(v2)
	out[1] = pMax.Clone().Add(pMin).Scale(1 - wa).Average(
		v1.Clone().Add(v2).Scale(wa),
	)
	out[2] = pMax.Clone().Scale(1 - wb).Add(vMax.Scale(wb))
	out[3] = pMin.Clone().Scale(1 - wb).Add(vMin.Scale(wb))

	// Figure out the best fit solution
	offspring, max := 0, float32(0.0)
	for i, o := range out {
		if f := o.Update(dst); f > max {
			offspring, max = i, f
		}
	}

	// Update with the best solution
	out[offspring].Update(dst)
}

// boundsOf computes the min, max chromosome bounds of the given population
func boundsOf(pop ...Evolver) (vector, vector) {
	n := len(pop[0].Chromosome())
	max := make(vector, n)
	min := make(vector, n)
	for i := 0; i < n; i++ {
		min[i] = math.MaxFloat32
	}

	for _, e := range pop {
		v := e.Chromosome()
		for i := 0; i < n; i++ {
			x := v[i]
			if x > max[i] {
				max[i] = x
			}
			if x < min[i] {
				min[i] = x
			}
		}
	}
	return min, max
}
