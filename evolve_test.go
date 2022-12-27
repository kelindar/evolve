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
	pop := newPop(256, target)

	// Evolve
	var last *binary.Genome
	for i := 0; i < 100000; i++ {
		if last = pop.Evolve(); last.String() == target {
			break
		}
	}

	assert.Equal(t, target, last.String())
}

func TestConverge(t *testing.T) {
	const target = "hello"
	const experiments = 100

	results := make([]float64, 0, experiments)
	for exp := 0; exp < experiments; exp++ {
		pop := newPop(256, target)
		for i := 0; i < 100000; i++ {
			if fittest := pop.Evolve(); fittest.String() == target {
				results = append(results, float64(i))
				break
			}
		}
	}

	assert.LessOrEqual(t, median(results), float64(125))
}

func TestRange(t *testing.T) {
	const target = "ab"
	pop := newPop(100, target)
	pop.Evolve()

	count := 0
	pop.Range(func(genome *binary.Genome, fitness float32) {
		count++
	})

	assert.Equal(t, 100, count)
}

// newPop returns a new population for tests
func newPop(n int, target string) *evolve.Population[*binary.Genome] {
	fit := fitnessFor(target)
	return evolve.New(n, fit, binary.New(len(target)))
}

// fitnessFor returns a fitness function for a string
func fitnessFor(text string) func(*binary.Genome) float32 {
	target := []byte(text)
	return func(genome *binary.Genome) float32 {
		var score float32
		for i, v := range *genome {
			if v == target[i] {
				score++
			}
		}
		return score / float32(len(target))
	}
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
