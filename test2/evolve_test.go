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
	const n = 10
	population := make([]Evolver, 0, n)
	for i := 0; i < n; i++ {
		population = append(population, newText(target))
	}

	for i := 0; i < 10000; i++ {
		assert.NoError(t, Evolve(population, CrossoverBestFit, MutateRandom))
	}

	sort.Slice(population, func(i, j int) bool {
		return population[i].Fitness() > population[j].Fitness()
	})
	assert.Equal(t, target, string(population[0].Chromosome()))

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
	value := make(Chromosome, len(target))
	for i := 0; i < len(value); i++ {
		value[i] = byte(rand.Intn(256))
	}

	return &text{
		value:  value,
		target: target,
	}
}

type text struct {
	value  Chromosome
	target string
}

func (t *text) Chromosome() Chromosome {
	return t.value
}

func (t *text) Fitness() float32 {
	fitness := 0.0
	target := t.target
	for i, v := range t.value {
		delta := math.Abs(float64(target[i]) - float64(v))
		//fitness -= math.Pow(delta, 2)
		fitness += 1 - delta/255
	}
	return float32(fitness)

	/*var score float32
	for i, v := range t.value {
		if v == t.target[i] {
			score++
		}
	}
	return score / float32(len(t.value))*/
}
