// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

import (
	"crypto/rand"
)

// Genome represents a genome string
type Genome []byte

// RandomGenome generates a random genome string
func randomGenome(length int) Genome {
	v := make(Genome, length)
	rand.Read(v)
	return v
}
