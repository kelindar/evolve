// Copyright (c) Roman Atachiants and contributors. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.

package main

import (
	"fmt"
	"math/rand"
	"os"
	"sync/atomic"
	"time"

	"github.com/itchyny/maze"
	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/neural"
)

var seed atomic.Int64

var (
	width  = 1
	height = 1
)

func main() {
	pop := evolve.New(256, evaluateMaze, func() *neural.Network {
		return neural.NewNetwork([]int{4, 8, 64, 8, 4})
	})

	var solved float64
	for i := 1; ; i++ { // loop forever
		fittest := pop.Evolve()
		fitness := evaluateMaze(fittest)

		// Every new population the maze will be different to avoid overfitting
		seed.Store(time.Now().UnixMicro())
		if fitness >= 95 {
			solved++
		}

		// Every 100 generations, reset and print out
		if i%100 == 0 {
			success := solved / 100.0 * 100
			m := createMaze(int(seed.Load()))
			solve(fittest, m)
			m.Print(os.Stdout, maze.Color)
			fmt.Printf("[#%.2d] level %d, success rate = %.2f%%\n", i, width, success)

			// If our success rate is high, consider the maze solved and increase the complexity
			// of the problem space
			switch {
			case success > 70:
				width += 1
				height += 1
			case success < 1:
				width -= 1
				height -= 1
			}

			// Reset the generation
			seed.Store(0)
			solved = 0
		}
	}
}

func evaluateMaze(g *neural.Network) float32 {
	return solve(g, createMaze(int(seed.Load())))
}

func solve(g *neural.Network, m *maze.Maze) (score float32) {
	sensor := make([]float32, 4)
	output := make([]float32, 4)

	// Clear the memory
	g.Reset()

	for n := 100; n > 0 && !m.Finished; n-- {
		out := g.Predict(sense(m, sensor), output)
		switch actionOf(out) {
		case 0:
			m.Move(maze.Up)
		case 1:
			m.Move(maze.Down)
		case 2:
			m.Move(maze.Left)
		case 3:
			m.Move(maze.Right)
		}

		// Reward more for shorter solutions
		if m.Finished {
			// return 90 + (10 * float32(n) / 100)
			return 100
		}
	}

	// This part rewards exploration of the maze by counting the visited cells
	visited := (maze.Up | maze.Down | maze.Left | maze.Right) << maze.VisitedOffset
	for x := range m.Directions {
		for y := range m.Directions[x] {
			if m.Directions[x][y] == visited {
				score += .5 // bonus points for exploration
			}
		}
	}

	return 0
}

func createMaze(seed int) *maze.Maze {
	rand.Seed(int64(seed))
	m := maze.NewMaze(width, height)
	m.Start = &maze.Point{X: 0, Y: 0}
	m.Goal = &maze.Point{X: height - 1, Y: width - 1}
	m.Cursor = m.Start
	m.Generate()
	return m
}

// sense goes through all neighbors and sets to 1 if the path is available
func sense(m *maze.Maze, vector []float32) (score []float32) {
	for i, direction := range maze.Directions {
		point := m.Cursor
		next := point.Advance(direction)
		if m.Contains(next) && m.Directions[point.X][point.Y]&direction == direction {
			vector[i] = 1.0
		}
	}
	return vector
}

// actionOf returns the best action to take (max value)
func actionOf(vector []float32) int {
	max, idx := vector[0], 0
	for i := 1; i < len(vector); i++ {
		if vector[i] > max {
			max = vector[i]
			idx = i
		}
	}
	return idx
}