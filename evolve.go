// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

import (
	"math/rand"
	"sync"
)

// Fitness represents a fitness function that evaluates a specific entity
type Fitness = func(Evolver) float32

// Genesis represents a function that creates a new genome
type Genesis = func() Genome

// Evolver represents an entity that evolves
type Evolver interface {
	Genome() Genome // Genome returns the genome
	Evolve(Genome)  // Evolve updates the genome
}

// Genome represents a genome contract.
type Genome interface {
	Mutate()
	Crossover(Genome, Genome)
}

// Population represents a population for evolution
type Population struct {
	sync.Mutex
	rand       *rand.Rand // The random number generator
	values     []Evolver  // The population that needs to be evolved
	fitnessFn  Fitness    // The fitness function
	fitnessMax float32    // The max fitness
	fitnessOf  []float32  // The fitness cache
	pool       int        // The current pool to use
	pools      [][]Genome // The genome pools to avoid allocs
}

// New creates a new population controller. This function takes a population of fixed
// size, a fitness function and a genome size (also of fixed size).
func New(population []Evolver, fitness Fitness, genesis Genesis) *Population {
	n := len(population) // Size of the population
	p := &Population{
		rand:      rand.New(rand.NewSource(1)),
		values:    population,
		pools:     make([][]Genome, 2),
		fitnessOf: make([]float32, n),
		fitnessFn: fitness,
	}

	// Create double-buffer for the genome strings
	p.pools[0] = make([]Genome, n)
	p.pools[1] = make([]Genome, n)
	for _, pool := range p.pools {
		for i := 0; i < n; i++ {
			pool[i] = genesis()
		}
	}

	// Write the initial random pool
	p.commit(p.pools[0])
	return p
}

// Commit writes the genome pool
func (p *Population) commit(pool []Genome) {
	for i, v := range p.values {
		v.Evolve(pool[i])
	}
}

// Evolve evolves the population
func (p *Population) Evolve() {
	p.Lock()
	defer p.Unlock()

	// Evaluate the fitness and cache it
	p.fitnessMax = p.evaluate()
	p.pool = (p.pool + 1) % 2
	buffer := p.pools[p.pool]
	for i := range p.values {

		// Select 2 parents
		p1, p2 := p.pickParents()

		// Perform the crossover
		gene := buffer[i]
		gene.Crossover(p1.Genome(), p2.Genome())

		// Mutate the genome
		gene.Mutate()
	}

	// Write the genome pool
	p.commit(buffer)
}

// pickParents selects 2 parents from the population and sorts them by their fitness.
func (p *Population) pickParents() (Evolver, Evolver) {
	p1, f1 := p.pickMate()
	p2, f2 := p.pickMate()
	if f1 > f2 {
		return p1, p2
	}

	return p2, p1
}

// pickMate selects a parent from the population
func (p *Population) pickMate() (Evolver, float32) {
	n := len(p.values)
	max := p.fitnessMax
	rng := p.rand
	for {
		i := rng.Intn(n)
		f := p.fitnessOf[i]
		if rng.Float32()*max <= f {
			return p.values[i], f
		}
	}
}

// evaluate computes the current fitness and caches it
func (p *Population) evaluate() (max float32) {
	for i, v := range p.values {
		f := p.fitnessFn(v)
		p.fitnessOf[i] = f
		if f > max {
			max = f
		}
	}
	return
}

// Fittest returns the fittest evolver
func (p *Population) Fittest() (best Evolver) {
	p.Lock()
	defer p.Unlock()

	max := float32(0)
	for i, v := range p.values {
		if f := p.fitnessOf[i]; f >= max {
			best, max = v, f
		}
	}

	return
}
