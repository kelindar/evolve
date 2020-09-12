// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

import (
	"math"
	"math/rand"
)

// Chromosome represents a gene chain
type Chromosome = []byte

// Mutate function mutates a chromosome in-place
type Mutate = func(Evolver)

// Evolver represents an entity that evolves
type Evolver interface {
	Chromosome() Chromosome // Chromosome returns the gene slice
	Fitness() float32       // Fitness evaluates the fitness
}

// Evolve evolves the population
func Evolve(pop []Evolver, crossover Crossover, mutate Mutate) error {

	// Perform the selection
	p1, p2, child := selection(pop)
	//p1, p2, child := ellitist(pop)

	// Perform the crossover and create an offspring
	crossover(p1, p2, child)

	// Mutate the population
	for _, v := range pop {
		mutate(v)
	}

	return nil
}

func ellitist(pop []Evolver) (Evolver, Evolver, Evolver) {
	min, max := float32(math.MaxFloat32), float32(0)
	p0, p1, p2 := 0, 0, 0
	for i := range pop {
		f := pop[i].Fitness()
		if f > max {
			max = f
			p2, p1 = p1, i
		}

		if f < min {
			min, p0 = f, i
		}
	}

	return pop[p1], pop[p2], pop[p0]
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
		if i1 < 0 && p1 <= sum {
			i1 = i
		}
		if i2 < 0 && p2 <= sum {
			i2 = i
		}

		// Done with the selection
		if i1 > 0 && i2 > 0 {
			break
		}
	}

	if i1 < 0 {
		i1 = 0
	}
	if i2 < 0 {
		i2 = 0
	}

	return pop[i1], pop[i2], weakestOf(pop)
}

// Selection selects 2 parents and an offspring destination
/*func selection(pop []Evolver) (Evolver, Evolver, Evolver) {
	total := totalFitness(pop)
	p1, p2 := rand.Float32(), rand.Float32()
	i1, i2 := -1, -1

	var sum float32
	for i, v := range pop {
		sum += v.Fitness() / total
		//fmt.Printf("%0.2f %0.2f %0.2f\n", sum, p1, p2)
		if i1 < 0 && p1 <= sum {
			i1 = i
		}
		if i2 < 0 && p2 <= sum {
			i2 = i
		}

		// Done with the selection
		if i1 > 0 && i2 > 0 {
			break
		}
	}

	if i1 < 0 {
		i1 = 0
	}
	if i2 < 0 {
		i2 = 0
	}

	return pop[i1], pop[i2], weakestOf(pop)
}*/

// maxFitness returns the max fitness
func maxFitness(pop []Evolver) (max float32) {
	for _, v := range pop {
		if f := v.Fitness(); f > max {
			max = f
		}
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
	/*if rand.Float32() > 0.01 {
		return
	}

	// Mutate a random gene
	c := dst.Chromosome()
	i := rand.Intn(len(c))
	v := rand.Intn(256)
	c[i] = byte(v)*/

	const rate = 0.01

	genes := dst.Chromosome()
	for i := 0; i < len(genes); i++ {
		if rand.Float32() <= rate {
			genes[i] = byte(rand.Intn(256))
		}
	}
}
