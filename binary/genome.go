// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package binary

import (
	crand "crypto/rand"
	mrand "math/rand"

	"github.com/kelindar/evolve"
)

// Genome represents a binary genome
type Genome []byte

// Crossover implements a random binary crossover
func (g *Genome) Crossover(p1, p2 evolve.Genome) {
	v1, v2 := *p1.(*Genome), *p2.(*Genome)
	dst := *g
	n := len(v1)
	for i := 0; i < n; i++ {
		r := randByte()
		dst[i] = (v1[i] & byte(r)) ^ (v2[i] & (^byte(r)))
	}
}

// Mutate mutates a random gene
func (g Genome) Mutate() {
	const rate = 0.01
	if mrand.Float32() >= rate {
		return
	}

	i := mrand.Int31n(int32(len(g)))
	g[i] = randByte()
}

// Make creates a function for a random genome string
func Make(length int) func() evolve.Genome {
	return func() evolve.Genome {
		v := make(Genome, length)
		crand.Read(v)
		return &v
	}
}

// randByte generates a random byte
func randByte() byte {
	return byte(mrand.Int31n(256))
}
