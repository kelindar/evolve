// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"sort"
	"sync/atomic"
)

var serial uint32

// Next generates a next sequence number.
func next() uint32 {
	return atomic.AddUint32(&serial, 1)
}

// ----------------------------------------------------------------------------------

type kind byte

const (
	isHidden kind = iota
	isInput
	isOutput
)

// ----------------------------------------------------------------------------------

// Neuron represents a neuron in the network
type neuron struct {
	Serial uint32    // The innovation serial number
	Kind   kind      // The neuron kind (hidden, input or output)
	Conns  []synapse // The incoming connections
	value  float64   // The output value (for activation)
}

// makeNeuron creates a new neuron.
func makeNeuron(kind kind) neuron {
	return neuron{
		Serial: next(),
		Kind:   kind,
	}
}

// connected checks whether the two neurons are connected or not.
func (n *neuron) connected(neuron *neuron) bool {
	return searchNode(n, neuron) || searchNode(neuron, n)
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

// makeNeurons creates a new neuron array.
func makeNeurons(inputs, outputs int) neurons {
	arr := make(neurons, 0, inputs+outputs)
	for i := 0; i < inputs; i++ {
		arr = append(arr, makeNeuron(isInput))
	}

	for i := 0; i < outputs; i++ {
		arr = append(arr, makeNeuron(isOutput))
	}
	return arr
}

// Len returns the number of neurons.
func (n neurons) Len() int {
	return len(n)
}

// Less compares two neurons in the slice.
func (n neurons) Less(i, j int) bool {
	return n[i].Serial < n[j].Serial
}

// Swap swaps two neurons.
func (n neurons) Swap(i, j int) {
	n[i], n[j] = n[j], n[i]
}

// Find searches the neurons for a specific serial. For this to work correctly,
// the neurons array should be sorted.
func (n neurons) Find(serial uint32) *neuron {
	i := sort.Search(len(n), func(i int) bool {
		return n[i].Serial >= serial
	})
	if i < len(n) && n[i].Serial == serial {
		return &n[i]
	}

	return nil
}

// Last selects the last neuron from the slice.
func (n neurons) Last() *neuron {
	return &n[len(n)-1]
}

// ----------------------------------------------------------------------------------

// Synapse represents a synapse for the NEAT network.
type synapse struct {
	Weight   float64 // The weight of the connection
	From, To *neuron // The neurons of the connection
	Active   bool    // Whether the connection is enabled or not
}

// ID returns a unique key for the edge.
func (s *synapse) ID() uint64 {
	return (uint64(s.To.Serial) << 32) | (uint64(s.From.Serial) & 0xffffffff)
}

// Equal checks whether the connection is equal to another connection
func (s *synapse) Equal(other *synapse) bool {
	return s.From == other.From && s.To == other.To
}

// ----------------------------------------------------------------------------------

// synapses represents a connection list which is sorted by neuron ID
type synapses []synapse

// Len returns the number of connections.
func (s synapses) Len() int {
	return len(s)
}

// Less compares two connections in the slice.
func (s synapses) Less(i, j int) bool {
	return s[i].ID() < s[j].ID()
}

// Swap swaps two connections
func (s synapses) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// Contains searches the synapses for a specific ID. For this to work correctly,
// the synapse array should be sorted.
func (s synapses) Contains(id uint64) (int, bool) {
	i := sort.Search(len(s), func(i int) bool {
		return s[i].ID() >= id
	})
	return i, i < len(s) && s[i].ID() == id
}
