// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEvolve(t *testing.T) {
	const target = "Oompa Loompa!"
	const n = 100
	population := make([]Evolver, 0, n)
	for i := 0; i < n; i++ {
		population = append(population, newText(target))
	}

	for i := 0; i < 1000; i++ {
		assert.NoError(t, Evolve(population, CrossoverAverageBound, MutateRandom))
	}

	sort.Slice(population, func(i, j int) bool {
		return population[i].Fitness() > population[j].Fitness()
	})
	assert.Equal(t, target, string(floatsToBytes(population[0].Chromosome())))

}

/*
func TestBinaryCrossover(t *testing.T) {
	v1 := newText("a")
	v2 := newText("a")
	v1.Chromosome()[0] = .5
	v2.Chromosome()[0] = .4

	v3 := newText("a")
	BinaryCrossover1(v1, v2, v3)
	assert.Equal(t, float32(0.503125), v3.Chromosome()[0])
}
*/
func newText(target string) Evolver {
	value := make([]float32, len(target))
	for i := 0; i < len(value); i++ {
		value[i] = rand.Float32()
	}

	return &text{
		value:  value,
		target: target,
	}
}

type text struct {
	value  []float32
	target string
}

func (t *text) Chromosome() []float32 {
	return t.value
}

func (t *text) Fitness() float32 {
	fitness := 0.0
	target := bytesToFloats([]byte(t.target))
	for i, v := range t.value {
		fitness += 10 - math.Abs(float64(target[i])-float64(v))*10
	}
	return float32(fitness)
}

// Converts a string to float
func bytesToFloats(v []byte) []float32 {
	out := make([]float32, 0, len(v))
	for _, c := range v {
		out = append(out, float32(c)/255)
	}
	return out
}

// Converts a string to float
func floatsToBytes(v []float32) []byte {
	out := make([]byte, 0, len(v))
	for _, c := range v {
		out = append(out, byte(c*255))
	}
	return out
}
