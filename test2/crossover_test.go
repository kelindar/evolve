// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package evolve

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCrossoverMidpoint(t *testing.T) {
	v1 := newText("a")
	v2 := newText("a")
	v1.Chromosome()[0] = 0b00111111
	v2.Chromosome()[0] = 0b11110011

	v3 := newText("a")
	CrossoverMidpoint1(v1, v2, v3)
	assert.Equal(t, byte(0b00110011), v3.Chromosome()[0])
}
