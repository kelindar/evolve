// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEvolve(t *testing.T) {
	const target = "This is evolving..."
	const n = 100
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
		if last = string(pop.best(fit).Genome()); last == target {
			break
		}
	}

	assert.Equal(t, 0, i)
	assert.Equal(t, target, string(pop.best(fit).Genome()))
}

type text struct {
	value Genome
}

func (t *text) Genome() Genome {
	return t.value
}

func (t *text) Evolve(v Genome) {
	t.value = v
}

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
