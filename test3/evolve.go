// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

import (
	"math/rand"
	"sync"
)

// Fitness represents a fitness function that evaluates a specific entity
type Fitness = func(Evolver) float32

// Evolver represents an entity that evolves
type Evolver interface {
	Genome() Genome // Genome returns the genome
	Evolve(Genome)  // Evolve updates the genome
}

// Population represents a population for evolution
type Population struct {
	sync.Mutex
	Values     []Evolver                        // The population that needs to be evolved
	mutate     func(Genome)                     // The mutation function
	crossover  func(p1, p2 Evolver, dst Genome) // The crossover function
	maxFitness float32                          // The max fitness
	pool       int                              // The current pool to use
	pools      [][]Genome                       // The genome pools to avoid allocs
}

// New creates a new population controller
func New(population []Evolver, length int) *Population {
	n := len(population) // Size of the population
	p := &Population{
		Values:    population,
		mutate:    MutateRandom,
		crossover: CrossoverBinary,
		pools:     make([][]Genome, 2),
	}

	// Create double-buffer for the genome strings
	p.pools[0] = make([]Genome, n)
	p.pools[1] = make([]Genome, n)
	for _, pool := range p.pools {
		for i := 0; i < n; i++ {
			pool[i] = randomGenome(length)
		}
	}

	// Write the initial random pool
	p.commit(p.pools[0])
	return p
}

// Commit writes the genome pool
func (p *Population) commit(pool []Genome) {
	for i, v := range p.Values {
		v.Evolve(pool[i])
	}
}

// Evolve evolves the population
func (p *Population) Evolve(fitness Fitness) {
	p.Lock()
	defer p.Unlock()

	// Calculate max fitness
	p.maxFitness = maxFitness(p.Values, fitness)
	p.pool = (p.pool + 1) % 2
	buffer := p.pools[p.pool]

	// Perform the selection & crossover
	for i := range p.Values {
		p1 := p.pickMate(fitness)
		p2 := p.pickMate(fitness)
		p.crossover(p1, p2, buffer[i])
	}

	// Mutate the population
	for _, v := range buffer {
		p.mutate(v)
	}

	// Write the genome pool
	p.commit(buffer)
}

// pickMate selects a parent from the population
func (p *Population) pickMate(fitnessOf Fitness) Evolver {
	n := len(p.Values)
	for {
		v := p.Values[rand.Intn(n)]
		r := rand.Float32() * p.maxFitness
		if r < fitnessOf(v) {
			return v
		}
	}
}

// maxFitness returns the max fitness
func maxFitness(pop []Evolver, fitnessOf Fitness) (max float32) {
	for _, v := range pop {
		if f := fitnessOf(v); f > max {
			max = f
		}
	}
	return
}

// best returns the best fit
func (p *Population) best(fitnessOf Fitness) (best Evolver) {
	max := float32(0)
	for _, v := range p.Values {
		f := fitnessOf(v)
		if f > max {
			best, max = v, f
		}
	}

	return
}

// MutateRandom ...
func MutateRandom(genes Genome) {
	const rate = 0.001

	for i := 0; i < len(genes); i++ {
		if rand.Float32() <= rate {
			genes[i] = randByte()
		}
	}
}
