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
func New(population []Evolver, fitness Fitness, genomeSize int) *Population {
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
			pool[i] = randomGenome(genomeSize)
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

	// Perform the selection & crossover
	for i := range p.values {
		p1 := p.pickMate()
		p2 := p.pickMate()
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
func (p *Population) pickMate() Evolver {
	n := len(p.values)
	max := p.fitnessMax
	for {
		i := rand.Intn(n)
		if p.rand.Float32()*max < p.fitnessOf[i] {
			return p.values[i]
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

// best returns the best fit
func (p *Population) best(fitnessOf Fitness) (best Evolver) {
	max := float32(0)
	for _, v := range p.values {
		f := fitnessOf(v)
		if f > max {
			best, max = v, f
		}
	}

	return
}

// mutate mutates a random gene
func (p *Population) mutate(genes Genome) {
	const rate = 0.01
	if p.rand.Float32() >= rate {
		return
	}

	i := p.rand.Int31n(int32(len(genes)))
	genes[i] = randByte()
}
