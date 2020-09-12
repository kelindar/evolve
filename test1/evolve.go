// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

import (
	"math"
	"math/rand"
)

// Chromosome represents a gene chain
type Chromosome = []float32

// Mutate function mutates a chromosome in-place
type Mutate = func(Evolver)

// Evolver represents an entity that evolves
type Evolver interface {
	Chromosome() Chromosome // Chromosome returns the gene slice
	Fitness() float32       // Fitness evaluates the fitness
}

// Evolve evolves the population
func Evolve(pop []Evolver, crossover Crossover, mutate Mutate) error {

	// Get the two best fit (p1, p2) indices in a single pass
	/*min, max := float32(math.MaxFloat32), float32(0)
	p0, p1, p2 := 0, 0, 0
	for i := range population {
		f := population[i].Fitness()
		if f > max {
			max = f
			p2, p1 = p1, i
		}

		if f < min {
			min, p0 = f, i
		}
	}*/

	// Perform the selection
	p1, p2, child := selection(pop)

	// Perform the crossover and create an offspring
	crossover(p1, p2, child, pop)

	// Mutate the population
	for _, v := range pop {
		mutate(v)
	}

	return nil
}

// Selection selects 2 parents and an offspring destination
func selection(pop []Evolver) (Evolver, Evolver, Evolver) {
	total := totalFitness(pop)
	p1, p2 := rand.Float32(), rand.Float32()
	i1, i2 := -1, -1

	var sum float32
	for i, v := range pop {
		sum += v.Fitness() / total
		//fmt.Printf("%0.2f %0.2f %0.2f\n", sum, p1, p2)
		if i1 < 0 && sum >= p1 {
			i1 = i
		}
		if i2 < 0 && sum >= p2 {
			i2 = i
		}

		// Done with the selection
		if i1 > 0 && i2 > 0 {
			break
		}
	}

	return pop[i1], pop[i2], weakestOf(pop)
}

// totalFitness returns the total fitness
func totalFitness(pop []Evolver) (sum float32) {
	for _, v := range pop {
		sum += v.Fitness()
	}
	return
}

// weakestOf returns the least fit
func weakestOf(pop []Evolver) Evolver {
	min, out := float32(math.MaxFloat32), 0
	for i := range pop {
		f := pop[i].Fitness()

		if f < min {
			min, out = f, i
		}
	}
	return pop[out]
}

// MutateRandom ...
func MutateRandom(dst Evolver) {
	if rand.Float32() > 0.01 {
		return
	}

	// Mutate a random gene
	c := dst.Chromosome()
	i := rand.Intn(len(c))
	v := rand.Float32()
	c[i] = v
}
