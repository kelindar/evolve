# Genetic Algorithm

![GitHub go.mod Go version](https://img.shields.io/github/go-mod/go-version/kelindar/evolve)
[![PkgGoDev](https://pkg.go.dev/badge/github.com/kelindar/evolve)](https://pkg.go.dev/github.com/kelindar/evolve)
[![Go Report Card](https://goreportcard.com/badge/github.com/kelindar/evolve)](https://goreportcard.com/report/github.com/kelindar/evolve)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Coverage Status](https://coveralls.io/repos/github/kelindar/evolve/badge.svg)](https://coveralls.io/github/kelindar/evolve)

This repository contains a simple implementation of a genetic algorithm for  evolving arbitrary `[]byte` genomes. Under the hood, it uses a simple random binary crossover and mutation to do the trick. There's a double-buffering in place to prevent unnecessary allocations and a relatively simple API around it.


## License

Tile is licensed under the [MIT License](LICENSE.md).