// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"math"
	"sort"
	"sync/atomic"
)

var serial uint32

// Next generates a next sequence number.
func next() uint32 {
	return atomic.AddUint32(&serial, 1)
}

// ----------------------------------------------------------------------------------

// Node represents a neuron in the network
type neuron struct {
	Serial uint32    // The innovation serial number
	Conns  []synapse // The incoming connections
	value  float64   // The output value (for activation)
}

// makeNeuron creates a new neuron.
func makeNode() neuron {
	return neuron{
		Serial: next(),
	}
}

// Value returns the value for the neuron
func (n *neuron) Value() float64 {
	if n.value != 0 || len(n.Conns) == 0 {
		return n.value
	}

	// Sum of the weighted inputs to the neuron
	s := 0.0
	for _, c := range n.Conns {
		if c.Active {
			s += c.Weight * c.From.Value()
		}
	}

	// Keep the value to avoid recalculating
	n.value = sigmoid(s)
	return n.value
}

// connected checks whether the two neurons are connected or not.
func (n *neuron) connected(neuron *neuron) bool {
	return searchNode(n, neuron) || searchNode(neuron, n)
}

// Sigmod activation function.
func sigmoid(x float64) float64 {
	return 1.0 / (1 + math.Exp(-x))
}

// searchNode searches whether incoming connections of "to" contain a "from" neuron.
func searchNode(from, to *neuron) bool {
	x := from.Serial
	i := sort.Search(len(to.Conns), func(i int) bool {
		return to.Conns[i].From.Serial >= x
	})
	return i < len(to.Conns) && to.Conns[i].From == from
}

// ----------------------------------------------------------------------------------

// Nodes represents a set of neurons
type neurons []neuron

// makeNodes creates a new neuron array.
func makeNodes(count int) neurons {
	arr := make(neurons, 0, count)
	for i := 0; i < count; i++ {
		arr = append(arr, makeNode())
	}
	return arr
}

// ----------------------------------------------------------------------------------

// Synapse represents a synapse for the NEAT network.
type synapse struct {
	Serial   uint32  // The innovation serial number
	Weight   float64 // The weight of the connection
	Active   bool    // Whether the connection is enabled or not
	From, To *neuron // The neurons of the connection
}

// ID returns a unique key for the edge.
func (c *synapse) ID() uint64 {
	return (uint64(c.To.Serial) << 32) | (uint64(c.From.Serial) & 0xffffffff)
}

// ----------------------------------------------------------------------------------

// sortedByNode represents a connection list which is sorted by neuron ID
type sortedByNode []synapse

// Len returns the number of connections.
func (c sortedByNode) Len() int {
	return len(c)
}

// Less compares two connections in the slice.
func (c sortedByNode) Less(i, j int) bool {
	return c[i].ID() < c[j].ID()
}

// Swap swaps two connections
func (c sortedByNode) Swap(i, j int) {
	c[i], c[j] = c[j], c[i]
}
