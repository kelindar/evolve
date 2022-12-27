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

	fit := fitnessFor(target)
	pop := evolve.New(n, fit, binary.New(len(target)))

	// Evolve
	var last *binary.Genome
	for i := 0; i < 100000; i++ {
		if last = pop.Evolve(); last.String() == target {
			break
		}
	}

	assert.Equal(t, target, last.String())
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
