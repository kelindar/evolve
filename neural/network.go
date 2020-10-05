// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package neural

import (
	"math/rand"
	"sort"
	"time"

	"github.com/kelindar/evolve"
)

// Network represents a neural network.
type Network struct {
	random *rand.Rand // Random source
	input  int        // Count of input neurons
	output int        // Count of output neurons
	nodes  neurons    // Neurons for the network
	conns  synapses   // Synapses, sorted by ID
}

// New creates a function for a random genome string
func New(in, out int) evolve.Genesis {
	origin := newNetwork(in, out)
	return func() evolve.Genome {
		clone := &Network{
			random: newRandom(),
		}

		origin.Clone(clone)
		return clone
	}
}

// newNetwork creates a new neural network.
func newNetwork(inputs, outputs int) *Network {
	nn := &Network{
		random: newRandom(),
		input:  inputs,
		output: outputs,
		nodes:  makeNeurons(1+inputs, outputs),
		conns:  make([]synapse, 0, 256),
	}

	// First is always a bias neuron
	nn.nodes[0].value = 1.0
	return nn
}

// Clone clones the neural network by copying all of the neurons and synapses and
// re-assigning the synapse pointers accordingly.
func (n *Network) Clone(dst *Network) {
	defer dst.sort()
	dst.clear()

	// Copy the nodes into the destination
	dst.input = n.input
	dst.output = n.output
	for _, v := range n.nodes {
		dst.nodes = append(dst.nodes, neuron{
			Serial: v.Serial,
			Kind:   v.Kind,
		})
	}

	// Sort the destination nodes so we can find the corresponding ones
	sort.Sort(dst.nodes)
	for _, v := range n.conns {
		dst.conns = append(dst.conns, synapse{
			From:   dst.nodes.Find(v.From.Serial),
			To:     dst.nodes.Find(v.To.Serial),
			Weight: v.Weight,
			Active: v.Active,
		})
	}
}

// Clear clears the network for reuse.
func (n *Network) clear() {
	n.nodes = n.nodes[:0]
	n.conns = n.conns[:0]
}

// sort sorts the connections depending on the neuron and assigns connection slices
// to the appropriate neurons for activation.
func (n *Network) sort() {
	if len(n.conns) == 0 {
		return
	}

	// Sort by neuron ID
	sort.Sort(n.conns)

	// Prepare the first connection
	conn := &n.conns[0]
	conn.From = n.nodes.Find(conn.From.Serial)
	conn.To = n.nodes.Find(conn.To.Serial)

	// Assign connection slices to neurons. This is basically sub-slicing the main
	// array, so the "Data" pointer of the slice will point to the same underlying
	// array, avoiding extra memory space and allocations.
	prev, lo := conn.To, 0
	curr, hi := conn.To, 0
	for i := 0; i < len(n.conns); i++ {
		conn = &n.conns[i]

		// Re-assign pointers, to make sure that the pointers to neurons are correct
		// which may be caused by nodes slice being re-allocated elsewhere during
		// append()
		conn.From = n.nodes.Find(conn.From.Serial)
		conn.To = n.nodes.Find(conn.To.Serial)
		curr, hi = conn.To, i
		if prev.Serial != curr.Serial {
			prev.Conns = n.conns[lo:hi]
			prev, lo = curr, hi
		}
	}

	// Last neuron
	prev.Conns = n.conns[lo : hi+1]
}

// connect connects two neurons together.
func (n *Network) connect(from, to uint32, weight float64) {
	n0 := n.nodes.Find(from)
	n1 := n.nodes.Find(to)
	if n0.connected(n1) {
		return
	}

	defer n.sort() // Keep sorted
	n.conns = append(n.conns, synapse{
		From:   n0,     // Left neuron
		To:     n1,     // Right neuron
		Weight: weight, // Weight for the connection
		Active: true,   // Default to active
	})
}

// Split splits the connetion by adding a neuron in the middle
func (n *Network) split(conn *synapse) {
	defer n.sort() // Keep sorted

	// Deactivate the connection and add a neuron
	conn.Active = false
	n.nodes = append(n.nodes, neuron{
		Serial: next(),
		Kind:   isHidden,
	})

	// Create 2 new connections
	middle := n.nodes.Last()
	n.connect(conn.From.Serial, middle.Serial, 1.0)
	n.connect(middle.Serial, conn.To.Serial, conn.Weight)
}

// Mutate mutates the network.
func (n *Network) Mutate() {
	defer n.sort() // Keep sorted

}

// Crossover applies genetic crossover between two networks. The first parent is
// the fittest of the two.
func (n *Network) Crossover(p1, p2 evolve.Genome) {

	// Copy of the nodes and synaposes from the fittest into this network
	n1, n2 := p1.(*Network), p2.(*Network)
	n1.Clone(n)

	// Iterate over the less fit parent and take matching genes randomly
	for _, conn := range n2.conns {
		if i, match := n.conns.Contains(conn.ID()); match {
			if n.random.Intn(2) == 0 {
				n.conns[i].Weight = conn.Weight
			}
		}
	}
}

// Distance calculates the distance between two neural networks based on their
// genome structure.
func (n *Network) Distance(other *Network) float64 {
	return 0
}

// newRandom creates a new random number generator
func newRandom() *rand.Rand {
	seed := time.Now().UnixNano()
	return rand.New(rand.NewSource(seed))
}
