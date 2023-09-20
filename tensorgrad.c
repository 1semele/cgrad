#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <stdarg.h>

#define MAX_DIMS 8

typedef enum {
	OP_NONE = 0,

	OP_ADD,
	OP_MUL,
} Op;

typedef struct {
	int ndim;
	int dim[MAX_DIMS];

	Op op;
	
	float *data;
} Tensor;

Tensor *tensor_create(int ndim, size_t space_needed, Op op) {
	Tensor *t = malloc(sizeof(Tensor));

	t->ndim = ndim;
	t->op = op;
	t->data = malloc(sizeof(float) * space_needed);

	return t;
}

Tensor *tensor_from_arr(const float *arr, size_t n) {
	Tensor *t = tensor_create(1, n, OP_NONE);
	t->dim[0] = n;
	memcpy(t->data, arr, sizeof(float) * n);
	return t;
}

void rand_rec(Tensor *t, int dim, int idx) {
	if (dim == t->ndim - 1) {
		for (int i = 0; i < t->dim[dim]; i++) {
			t->data[idx + i] = (float)rand()/(float)(RAND_MAX);
		}
	} else {
		for (int i = 0; i < t->dim[dim]; i++) {
			rand_rec(t, dim + 1, idx + t->dim[dim] * i);
		}
	}
}

Tensor *tensor_rand(int ndim, int *in_dim) {
	int space_needed = 1;

	for (int i = 0; i < ndim; i++) {
		space_needed *= in_dim[i];
	}

	Tensor *t = tensor_create(ndim, space_needed, OP_NONE);
	for (int i = 0; i < ndim; i++) {
		t->dim[i] = in_dim[i];
	}

	rand_rec(t, 0, 0);

	return t;
}

void value_rec(Tensor *t, float val, int dim, int idx) {
	if (dim == t->ndim - 1) {
		for (int i = 0; i < t->dim[dim]; i++) {
			t->data[idx + i] = val;
		}
	} else {
		for (int i = 0; i < t->dim[dim]; i++) {
			value_rec(t, val, dim + 1, idx + t->dim[dim] * i);
		}
	}
}

Tensor *tensor_value(float val, int ndim, int *in_dim) {
	int space_needed = 1;

	for (int i = 0; i < ndim; i++) {
		space_needed *= in_dim[i];
	}

	Tensor *t = tensor_create(ndim, space_needed, OP_NONE);
	for (int i = 0; i < ndim; i++) {
		t->dim[i] = in_dim[i];
	}

	value_rec(t, val, 0, 0);

	return t;
}

void print_indent(int level) {
	for (int i = 0; i < level; i++) {
		printf("  ");
	}
}

void print_rec(Tensor *t, int dim, int idx) {
	if (dim == t->ndim - 1) {
		print_indent(dim);
		printf("[");
		for (int i = 0; i < t->dim[dim] - 1; i++) {
			printf("%f, ", t->data[idx + i]);
		}
		printf("%f]\n", t->data[idx + t->dim[dim] - 1]);
	} else {
		print_indent(dim);
		printf("[\n");
		for (int i = 0; i < t->dim[dim]; i++) {
			print_rec(t, dim + 1, idx + t->dim[dim] * i);
		}
		print_indent(dim);
		printf("]\n");
	}
}


void tensor_print(Tensor *t) {
	print_rec(t, 0, 0);
}

void tensor_print_shape(Tensor *t) {
	printf("[");
	for (int i = 0; i < t->ndim - 1; i++) {
		printf("%d, ", t->dim[i]);
	}
	printf("%d]\n", t->dim[t->ndim - 1]);
}

bool tensor_check_size(Tensor *t1, Tensor *t2) {
	if (t1->ndim != t2->ndim) {
		return false;
	}

	for (int i = 0; i < t1->ndim; i++) {
		if (t1->dim[i] != t2->dim[i]) {
			return false;
		}
	}

	return true;
}

void tensor_copy_dim(Tensor *src, Tensor *dst) {
	for (int i = 0; i < src->ndim; i++) {
		dst->dim[i] = src->dim[i];
	}
}

int tensor_size(Tensor *t) {
	int size = 1;
	for (int i = 0; i < t->ndim; i++) {
		size *= t->dim[i];
	}
	return size;
}

void add_rec(Tensor *t1, Tensor *t2, Tensor *out, int dim, int idx) {
	if (dim == t1->ndim - 1) {
		for (int i = 0; i < t1->dim[dim]; i++) {
			out->data[idx + i] = t1->data[idx + i] + t2->data[idx + i];
		}
	}
	else {
		for (int i = 0; i < t1->dim[dim]; i++) {
			add_rec(t1, t2, out, dim + 1, idx + t1->dim[dim] * i);
		}
	}
}

Tensor *tensor_add(Tensor *t1, Tensor *t2) {
	if (!tensor_check_size(t1, t2)) {
		assert(0);
	}

	int size = tensor_size(t1);
	Tensor *out = tensor_create(t1->ndim, size, OP_ADD);
	tensor_copy_dim(t1, out);
	
	add_rec(t1, t2, out, 0, 0);

	return out;
}

void mul_rec(Tensor *t1, Tensor *t2, Tensor *out, int dim, int idx) {
	if (dim == t1->ndim - 1) {
		for (int i = 0; i < t1->dim[dim]; i++) {
			out->data[idx + i] = t1->data[idx + i] * t2->data[idx + i];
		}
	}
	else {
		for (int i = 0; i < t1->dim[dim]; i++) {
			mul_rec(t1, t2, out, dim + 1, idx + t1->dim[dim] * i);
		}
	}
}

Tensor *tensor_mul(Tensor *t1, Tensor *t2) {
	if (!tensor_check_size(t1, t2)) {
		assert(0);
	}

	int size = tensor_size(t1);
	Tensor *out = tensor_create(t1->ndim, size, OP_MUL);
	tensor_copy_dim(t1, out);
	
	mul_rec(t1, t2, out, 0, 0);

	return out;
}

int main() {
	srand(1);
	
	int shape[] = {
		2, 3
	};

	Tensor *t1 = tensor_value(3.0, 2, shape);
	Tensor *t2 = tensor_value(4.0, 2, shape);
	Tensor *t3 = tensor_value(5.0, 2, shape);
	Tensor *t4 = tensor_mul(t1, t2);
	Tensor *t5 = tensor_add(t3, t4);


	tensor_print_shape(t5);

	tensor_print(t5);

	return EXIT_SUCCESS;
}
