package main

import (
	"fmt"
	"math/rand"
	"os"
	"sync/atomic"

	"github.com/itchyny/maze"
	"github.com/kelindar/evolve"
	"github.com/kelindar/evolve/ffnet"
)

var seed atomic.Int64

var (
	width  = 2
	height = 2
)

func main() {
	pop := evolve.New(256, evaluateMaze, func() *ffnet.FeedForward {
		return ffnet.NewFeedForward([]int{4, 8, 8, 8, 4})
	})

	for i := 0; ; i++ { // loop forever
		fittest := pop.Evolve()
		fitness := evaluateMaze(fittest)
		success := float64(seed.Load()) / float64(i) * 100

		// If we solved the maze, change the shape
		if fitness == 100 {
			seed.Add(1)
		}

		// Every 1000 generations, reset and print out
		if i%1000 == 0 {
			m := createMaze(int(seed.Load()))
			m.Print(os.Stdout, maze.Color)
			fmt.Printf("[#%.2d] best score = %.2f%%, success rate = %.2f%%\n", i, fitness, success)
			///fmt.Println(fittest.String())

			// If our success rate is high, consider the maze solved and increase the complexity
			// of the problem space
			if success > 90 {
				width += 1
				height += 1
			}

			// Reset the generation
			seed.Store(0)
			i = 0
		}
	}
}

func evaluateMaze(g *ffnet.FeedForward) (score float32) {
	m := createMaze(int(seed.Load()))
	sensor := make([]float32, 4)
	output := make([]float32, 4)

	var exploration float32
	for score = 100; score > 0; score -= 1 {
		exploration += sense(m, sensor)

		out := g.Predict(sensor, output)
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

		if m.Finished {
			return 100
		}
	}

	score += float32(exploration)
	return
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
func sense(m *maze.Maze, vector []float32) (score float32) {
	for i, direction := range maze.Directions {
		point := m.Cursor
		next := point.Advance(direction)
		if m.Contains(next) && m.Directions[point.X][point.Y]&direction == direction {
			vector[i] = 1.0
			//score += 0.2
		}
	}
	return score
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
