// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEvolve(t *testing.T) {
	const target = "This is evolving..."
	const n = 200
	population := make([]Evolver, 0, n)
	for i := 0; i < n; i++ {
		population = append(population, new(text))
	}

	fit := fitnessFor(target)
	pop := New(population, fit, len(target))

	// Evolve
	i, last := 0, ""
	for ; i < 100000; i++ {
		pop.Evolve()
		//	println(string(pop.best(fit).Genome()))
		if last = string(pop.Fittest().Genome()); last == target {
			break
		}
	}

	assert.Equal(t, target, string(pop.Fittest().Genome()))
}

// fitnessFor returns a fitness function for a string
func fitnessFor(text string) Fitness {
	target := []byte(text)
	return func(v Evolver) float32 {
		var score float32
		for i, v := range v.Genome() {
			if v == target[i] {
				score++
			}
		}
		return score / float32(len(target))
	}
}

// Text represents a text with a dna (text itself in this case)
type text struct {
	dna Genome
}

// Genome returns the genome
func (t *text) Genome() Genome {
	return t.dna
}

// Evolve updates the genome
func (t *text) Evolve(v Genome) {
	t.dna = v
}

// String returns a string representation
func (t *text) String() string {
	return string(t.dna)
}
