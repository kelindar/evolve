// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package binary_test

import (
	"testing"

	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/binary"
	"github.com/stretchr/testify/assert"
)

func TestEvolve(t *testing.T) {
	const target = "abc"
	const n = 200
	population := make([]evolve.Evolver, 0, n)
	for i := 0; i < n; i++ {
		population = append(population, new(text))
	}

	fit := fitnessFor(target)
	pop := evolve.New(population, fit, binary.New(len(target)))

	// Evolve
	var last evolve.Evolver
	for i := 0; i < 100000; i++ {
		if last = pop.Evolve(); toString(last.Genome()) == target {
			break
		}
	}

	assert.Equal(t, target, toString(last.Genome()))
}

// fitnessFor returns a fitness function for a string
func fitnessFor(text string) evolve.Fitness {
	target := []byte(text)
	return func(v evolve.Evolver) float32 {
		var score float32
		genome := v.Genome().(*binary.Genome)
		for i, v := range *genome {
			if v == target[i] {
				score++
			}
		}
		return score / float32(len(target))
	}
}

// Text represents a text with a dna (text itself in this case)
type text struct {
	dna evolve.Genome
}

// Genome returns the genome
func (t *text) Genome() evolve.Genome {
	return t.dna
}

// Evolve updates the genome
func (t *text) Evolve(v evolve.Genome) {
	t.dna = v
}

func toString(v evolve.Genome) string {
	return string(*v.(*binary.Genome))
}
