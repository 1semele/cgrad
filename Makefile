all: scalargrad

scalargrad: scalargrad.c
	gcc -o scalargrad scalargrad.c -Wall -Wextra -lm

run: scalargrad
	@./scalargrad
