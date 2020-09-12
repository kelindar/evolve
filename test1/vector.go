// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

// Vector represents a vector for vector math
type vector []float32

// Clone copies the vector
func (v vector) Clone() vector {
	out := make(vector, len(v))
	for i := range v {
		out[i] = v[i]
	}
	return out
}

// Add adds two vectors together
func (v vector) Add(b vector) vector {
	for i := range v {
		v[i] = v[i] + b[i]
	}
	return v
}

// Scale multiplies a vector by a scalar
func (v vector) Scale(scale float32) vector {
	for i := range v {
		v[i] = v[i] * scale
	}
	return v
}

// Average computes the average vector
func (v vector) Average(b vector) vector {
	for i := range v {
		v[i] = (v[i] + b[i]) / 2
	}
	return v
}

// Update updates the destination and returns the fitness
func (v vector) Update(dst Evolver) float32 {
	out := dst.Chromosome()
	for i := 0; i < len(out); i++ {
		out[i] = v[i]
	}
	return dst.Fitness()
}
