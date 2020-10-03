// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestClone(t *testing.T) {
	serial = 0
	create := New(2, 1)
	nn1 := create().(*Network)
	nn2 := create().(*Network)

	// Neurons must have ascending serial number
	assert.Equal(t, uint32(1), nn1.nodes[0].Serial)
	assert.Equal(t, uint32(2), nn1.nodes[1].Serial)
	assert.Equal(t, uint32(3), nn1.nodes[2].Serial)
	assert.Equal(t, uint32(4), nn1.nodes[3].Serial)

	// Must be sorted
	assert.True(t, sort.IsSorted(nn1.nodes))

	// Add a connection and clone
	nn1.connect(&nn1.nodes[1], &nn1.nodes[3], 0.5)
	nn1.Clone(nn2)

	// Clone must match
	assert.Exactly(t, nn1.nodes, nn2.nodes)
	assert.Exactly(t, nn1.conns, nn2.conns)

	// Pointers must differ
	assert.True(t, nn1.conns[0].From != nn2.conns[0].From)
	assert.True(t, nn1.conns[0].To != nn2.conns[0].To)
}

// make2x2 creates a 2x2 tiny network
func make2x2() *Network {
	nn := New(2, 2)().(*Network)
	i0 := &nn.nodes[1]
	i1 := &nn.nodes[2]
	o0 := &nn.nodes[3]
	o1 := &nn.nodes[4]

	// connect inputs to output
	nn.connect(i0, o0, .5)
	nn.connect(i1, o0, .5)
	nn.connect(i0, o1, .75)
	return nn
}
