// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

import (
	"math/rand"
	"runtime"
	"sync"
)

// Genome represents a genome contract.
type Genome interface {
	Crossover(Genome, Genome)
	Mutate()
	Reset()
}

// Population represents a population for evolution
type Population[T Genome] struct {
	mu        sync.RWMutex
	rand      *rand.Rand      // The random number generator
	fitnessOf []float32       // The fitness cache
	fitnessFn func(T) float32 // The fitness function
	genomes   []T             // The current pool
	pool      int             // The current pool index
	pools     [2][]T          // The genome pools to avoid allocs
}

// New creates a new population controller. This function takes a population of fixed
// size, a fitness function and a genome size (also of fixed size).
func New[T Genome](n int, fitness func(T) float32, genesis func() T) *Population[T] {
	p := &Population[T]{
		rand:      rand.New(rand.NewSource(1)),
		pools:     [2][]T{},
		fitnessOf: make([]float32, n),
		fitnessFn: fitness,
	}

	// Create double-buffer for the genome strings
	p.pools[0] = make([]T, n)
	p.pools[1] = make([]T, n)
	for _, pool := range p.pools {
		for i := 0; i < n; i++ {
			pool[i] = genesis()
		}
	}

	p.genomes = p.pools[0]
	return p
}

// Range iterates over the current set of genomes and their fitness
func (p *Population[T]) Range(fn func(genome T, fitness float32)) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	for i, genome := range p.genomes {
		fn(genome, p.fitnessOf[i])
	}
}

// Evolve evolves the population
func (p *Population[T]) Evolve() (fittest T) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Parallelize the fitness evaluation
	p.evaluate(runtime.NumCPU())

	// Find the fittest genome
	best := float32(0)
	for i := range p.genomes {
		if fitness := p.fitnessOf[i]; fitness >= best {
			fittest = p.genomes[i]
			best = fitness
		}
	}

	p.pool = (p.pool + 1) % 2
	buffer := p.pools[p.pool]
	for i := range p.genomes {

		// Select 2 parents
		p1, p2 := p.pickParents()

		// Perform the crossover
		gene := buffer[i]
		gene.Crossover(p1, p2)

		// Mutate the genome
		gene.Mutate()
	}

	// Write the genome pool
	p.genomes = buffer
	return
}

// pickParents selects 2 parents from the population and sorts them by their fitness.
func (p *Population[T]) pickParents() (T, T) {
	p1, f1 := p.pickMate()
	p2, f2 := p.pickMate()
	if f1 > f2 {
		return p1, p2
	}

	return p2, p1
}

// pickMate selects a parent from the population using a tournament selection.
func (p *Population[T]) pickMate() (bestEvolver T, bestFitness float32) {
	const tournamentSize = 4
	for r := 0; r < tournamentSize; r++ {
		i := p.rand.Int31n(int32(len(p.genomes)))
		if f := p.fitnessOf[i]; f >= bestFitness || bestFitness == 0 {
			bestEvolver = p.genomes[i]
			bestFitness = f
		}
	}
	return
}

// evaluate evaluates the population in parallel
func (p *Population[T]) evaluate(parallelism int) {
	chunkSize := len(p.genomes) / parallelism
	var wg sync.WaitGroup
	wg.Add(parallelism)

	// launch a Goroutine for each chunk
	for i := 0; i < parallelism; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if i == parallelism-1 {
			end = len(p.genomes)
		}

		go func(start, end int) {
			for j := start; j < end; j++ {
				v := p.genomes[j]
				v.Reset()

				// Evaluate the fitness
				fitness := p.fitnessFn(v)
				p.fitnessOf[j] = fitness
			}
			wg.Done()
		}(start, end)
	}

	wg.Wait()
}
