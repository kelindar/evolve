// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConnected(t *testing.T) {
	n := makeNodes(2)
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
