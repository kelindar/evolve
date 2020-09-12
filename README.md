# Genetic Algorithm

![GitHub go.mod Go version](https://img.shields.io/github/go-mod/go-version/kelindar/evolve)
[![PkgGoDev](https://pkg.go.dev/badge/github.com/kelindar/evolve)](https://pkg.go.dev/github.com/kelindar/evolve)
[![Go Report Card](https://goreportcard.com/badge/github.com/kelindar/evolve)](https://goreportcard.com/report/github.com/kelindar/evolve)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Coverage Status](https://coveralls.io/repos/github/kelindar/evolve/badge.svg)](https://coveralls.io/github/kelindar/evolve)

This repository contains a simple implementation of a genetic algorithm for  evolving arbitrary `[]byte` genomes. Under the hood, it uses a simple random binary crossover and mutation to do the trick. There's a double-buffering in place to prevent unnecessary allocations and a relatively simple API around it.


## Usage

In order to use this, we first need to create a "phenotype" representation which contains the dna `[]byte`. It should implement the `Evolver` interface which contains `Genome()` and `Evolve()` methods, in the example here we are creating a simple text which contains the binary representation of the text itself.

```
// Text represents a text with a dna (text itself in this case)
type text struct {
	dna []byte
}

// Genome returns the genome
func (t *text) Genome() []byte {
	return t.dna
}

// Evolve updates the genome
func (t *text) Evolve(v []byte) {
	t.dna = v
}

// String returns a string representation
func (t *text) String() string {
	return string(t.dna)
}
```
Next, we'll need a fitness function to evaluate how good a genome is. In this example we're creating a fitness function for an abritrary string which simply returns a `func(Evolver) float32`
```

// fitnessFor returns a fitness function for a string
func fitnessFor(text string) evolve.Fitness {
	target := []byte(text)
	return func(v evolve.Evolver) float32 {
		var score float32
		for i, v := range v.Genome() {
			if v == target[i] {
				score++
			}
		}
		return score / float32(len(target))
	}
}
```

Finally, we can wire everything together by using `New()` function to create a population, and evolve it by repeatedly calling `Evolve()` method as shown below.

```
func main() {
	const target = "Hello World"
	const n = 200

    // Create a fitness function
	fit := fitnessFor(target)

    // Create a population
	population := make([]evolve.Evolver, 0, n)
	for i := 0; i < n; i++ {
		population = append(population, new(text))
	}
    
    // Create a population
	pop := evolve.New(population, fit, len(target))

	// Evolve over many generations
	for i := 0 ; i < 100000; i++ {
		pop.Evolve()
	}

    // Get the fittest member of the population
	fittest := pop.Fittest()
}
```

## License

Tile is licensed under the [MIT License](LICENSE.md).