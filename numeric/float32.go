// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package numeric

import (
	crand "crypto/rand"
	"encoding/binary"
	"fmt"
	"math"
	mrand "math/rand"

	"github.com/kelindar/evolve"
)

// Float32s represents a float32 numeric genome
type Float32s []float32

// New creates a function for a random genome string
func New(length int) func() *Float32s {
	return func() *Float32s {
		result := make(Float32s, length)
		for i := 0; i < length; i++ {
			result[i] = randFloat32()
		}
		return &result
	}
}

// String implement stringer interface
func (g *Float32s) String() string {
	if g == nil {
		return "<nil>"
	}

	return fmt.Sprintf("%+v", *g)
}

// Mutate mutates a random gene
func (g *Float32s) Mutate() {
	const rate = 0.01
	if mrand.Float32() >= rate {
		return
	}

	i := mrand.Int31n(int32(len(*g)))
	(*g)[i] = mrand.Float32()
}

// Crossover implements a random binary crossover
func (g *Float32s) Crossover(p1, p2 evolve.Genome) {
	v1, v2 := *p1.(*Float32s), *p2.(*Float32s)
	n := len(v1)
	for i := 0; i < n; i++ {
		(*g)[i] = crossover(v1[i], v2[i])
	}
}

// crossover calculates a crossover between 2 numbers
func crossover(v1, v2 float32) float32 {
	const delta = 0.5
	switch {
	case isNan(v1) && isNan(v2):
		return randFloat32()
	case isNan(v1):
		return v2
	case isNan(v2) || v1 == v2:
		return v1
	default: // e.g. [5, 10], move by x% towards 10
		return v1 + ((v2 - v1) * delta)
	}
}

func isNan(v float32) bool {
	return v != v
}

func randFloat32() float32 {
	v := make([]byte, 4)
	crand.Read(v)
	return math.Float32frombits(binary.BigEndian.Uint32(v))
}

func abs32(x float32) float32 {
	return math.Float32frombits(math.Float32bits(x) &^ (1 << 31))
}
