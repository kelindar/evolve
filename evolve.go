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
	mu        sync.Mutex
	rand      *rand.Rand  // The random number generator
	values    []Evolver   // The population that needs to be evolved
	fitnessOf []float32   // The fitness cache
	fitnessFn Fitness     // The fitness function
	pool      int         // The current pool to use
	pools     [2][]Genome // The genome pools to avoid allocs
}

// New creates a new population controller. This function takes a population of fixed
// size, a fitness function and a genome size (also of fixed size).
func New(population []Evolver, fitness Fitness, genesis Genesis) *Population {
	n := len(population) // Size of the population
	p := &Population{
		rand:      rand.New(rand.NewSource(1)),
		values:    population,
		pools:     [2][]Genome{},
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
func (p *Population) Evolve() (fittest Evolver) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Evaluate the fitness and cache it
	best := float32(0)
	for i, v := range p.values {
		p.fitnessOf[i] = p.fitnessFn(v)
		if fitness := p.fitnessOf[i]; fitness >= best {
			fittest = p.values[i]
			best = fitness
		}
	}

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
	return
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

// pickMate selects a parent from the population using a tournament selection.
func (p *Population) pickMate() (bestEvolver Evolver, bestFitness float32) {
	const tournamentSize = 4
	for r := 0; r < tournamentSize; r++ {
		i := p.rand.Int31n(int32(len(p.values)))
		if f := p.fitnessOf[i]; f >= bestFitness {
			bestEvolver = p.values[i]
			bestFitness = f
		}
	}
	return
}
