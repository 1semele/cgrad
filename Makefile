all: scalargrad tensorgrad

scalargrad: scalargrad.c
	gcc -o scalargrad scalargrad.c -Wall -Wextra -lm

tensorgrad: tensorgrad.c
	gcc -o tensorgrad tensorgrad.c -Wall -Wextra -lm

run: tensorgrad
	@./tensorgrad
