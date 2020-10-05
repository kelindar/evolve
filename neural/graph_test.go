// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConnected(t *testing.T) {
	n := []neuron{
		makeNeuron(isHidden),
		makeNeuron(isHidden),
	}

	n0, n1 := &n[0], &n[1]

	// Disjoint
	assert.False(t, n0.connected(n1))
	assert.False(t, n1.connected(n0))

	// Connect
	n1.Conns = append(n1.Conns, synapse{
		From: n0,
		To:   n1,
	})

	// Connected
	assert.True(t, n0.connected(n1))
	assert.True(t, n1.connected(n0))
}

func TestSplit(t *testing.T) {
	serial = 0
	nn := New(1, 1)().(*Network)
	in := &nn.nodes[1]
	out := &nn.nodes[2]

	// create a hidden layer
	for i := 0; i < 10; i++ {
		nn.split(&synapse{
			From:   in,
			To:     out,
			Weight: 0.5,
		})
	}

	/*for _, n := range nn.nodes {
		println("neuron", n.Serial, len(n.Conns))
		for _, c := range n.Conns {
			println("  ", c.From.Serial, "->", c.To.Serial)
		}
	}*/

	assert.Equal(t, 0, len(nn.nodes[1].Conns))
	assert.Equal(t, 10, len(nn.nodes[2].Conns))
	for i := 3; i < 13; i++ {
		assert.Equal(t, 1, len(nn.nodes[i].Conns))
	}
}
