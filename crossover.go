// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

import (
	"github.com/kelindar/rand"
)

// randByte generates a random byte
func randByte() byte {
	return byte(rand.Uint32n(256))
}

// crossover implements a random binary crossover
func (p *Population) crossover(p1, p2 Evolver, dst Genome) {
	v1, v2 := p1.Genome(), p2.Genome()

	n := len(v1)
	for i := 0; i < n; i++ {
		r := randByte()
		dst[i] = (v1[i] & byte(r)) ^ (v2[i] & (^byte(r)))
	}
}
