// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve_test

import (
	"sort"
	"testing"

	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/binary"
	"github.com/stretchr/testify/assert"
)

/*
cpu: Intel(R) Core(TM) i7-9700K CPU @ 3.60GHz
BenchmarkGenome/evolve-8         	   14208	     84960 ns/op	       0 B/op	       0 allocs/op
*/
func BenchmarkGenome(b *testing.B) {
	const target = "This is evolving..."

	b.Run("evolve", func(b *testing.B) {
		pop := newPop(256, target)
		b.ResetTimer()
		b.ReportAllocs()
		for n := 0; n < b.N; n++ {
			pop.Evolve()
		}
	})
}

func TestEvolve(t *testing.T) {
	const target = "This is evolving..."
	const n = 200
	pop := newPop(256, target)

	// Evolve
	var last evolve.Evolver
	for i := 0; i < 100000; i++ {

		//	println(string(pop.best(fit).Genome()))
		if last = pop.Evolve(); toString(last.Genome()) == target {
			break
		}
	}

	assert.Equal(t, target, toString(last.Genome()))
}

func TestConverge(t *testing.T) {
	const target = "hello"
	const experiments = 100

	results := make([]float64, 0, experiments)
	for exp := 0; exp < experiments; exp++ {
		pop := newPop(256, target)
		for i := 0; i < 100000; i++ {
			if fittest := pop.Evolve(); toString(fittest.Genome()) == target {
				results = append(results, float64(i))
				break
			}
		}
	}

	assert.LessOrEqual(t, median(results), float64(125))
}

// newPop returns a new population for tests
func newPop(n int, target string) *evolve.Population {
	population := make([]evolve.Evolver, 0, n)
	for i := 0; i < n; i++ {
		population = append(population, new(text))
	}

	fit := fitnessFor(target)
	return evolve.New(population, fit, binary.New(len(target)))
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

func median(data []float64) float64 {
	dataCopy := make([]float64, len(data))
	copy(dataCopy, data)

	sort.Float64s(dataCopy)

	var median float64
	l := len(dataCopy)
	if l == 0 {
		return 0
	} else if l%2 == 0 {
		median = (dataCopy[l/2-1] + dataCopy[l/2]) / 2
	} else {
		median = dataCopy[l/2]
	}

	return median
}
