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
} Shape;

/* Iterates through a shapes indexes in row first order. */
typedef struct {
    int idx[MAX_DIMS];
    Shape *sh;
    bool first;
} Shape_Iter;

typedef struct {
    Shape sh;
	int stride[MAX_DIMS];

	Op op;
	
	float *data;
} Tensor;

typedef struct {
	int idx[MAX_DIMS];
	Tensor *t;
    int full_idx;
} Tensor_Iter;

void shape_copy(Shape *src, Shape *dst) {
    dst->ndim = src->ndim;
    for (int i = 0; i < dst->ndim; i++) {
        dst->dim[i] = src->dim[i];
    }
}

int shape_space_needed(Shape *sh) {
    int space = 1;
    
    for (int i = 0; i < sh->ndim; i++) {
        space *= sh->dim[i];
    }

    return space;
}

Shape_Iter shape_iter_create(Shape *sh) {
    Shape_Iter sh_iter = {
        .sh = sh,
        .first = true,
    };

    for (int i = 0; i < sh->ndim; i++) {
        sh_iter.idx[i] = 0;
    }

    return sh_iter;
}

/* This can be MUCH shorter, but it's functional for now. */
bool shape_iter_next(Shape_Iter *sh_iter) {
    Shape *sh = sh_iter->sh;

    if (sh_iter->first) {
        sh_iter->first = false;
        return true;
    }

    int last_dim = sh->ndim - 1;
    sh_iter->idx[last_dim]++;

    if (sh_iter->idx[last_dim] == sh->dim[last_dim]) {
        if (last_dim == 0) {
            return false;
        }

        sh_iter->idx[last_dim] = 0;
        int dim = last_dim - 1;
        while (dim != -1) {
            if (sh_iter->idx[dim]++ == sh->dim[dim] - 1) {
                sh_iter->idx[dim] = 0;
                dim--;
                continue;
            }
            else {
                return true;
            }
        }
        return false;
    } 
    
    return true;
}

Tensor *tensor_create(Shape *sh, Op op) {
	Tensor *t = malloc(sizeof(Tensor));

    shape_copy(sh, &t->sh);
    int space_needed = shape_space_needed(sh);

    int stride = 1;
    for (int i = sh->ndim - 1; i >= 0; i--) {
        t->stride[i] = stride;
        stride *= sh->dim[i];
    }

	t->op = op;
	t->data = malloc(sizeof(float) * space_needed);

	return t;
}

int tensor_compute_idx(Tensor *t, int *idx) {
    int full_idx = 0;
    for (int i = 0; i < t->sh.ndim; i++) {
        full_idx += idx[i] * t->stride[i];
    }
    return full_idx;
}

Tensor *tensor_arange(int start, int end) {
    int len = end - start;
    Shape sh = {
        .ndim = 1,
        .dim = {len},
    };

    Tensor *t = tensor_create(&sh, OP_NONE);

    /* Fill in our actual data. */
    for (int i = 0; i < len; i++) {
        t->data[i] = i + start;
    }

    return t;
}

void print_indent(int level) {
	for (int i = 0; i < level; i++) {
		printf("  ");
	}
}

void tensor_print(Tensor *t) {
    Shape_Iter sh_iter = shape_iter_create(&t->sh);
    while (shape_iter_next(&sh_iter)) {
        for (int i = 0; i < t->sh.ndim; i++) {
            if (sh_iter.idx[i] == 0) {
                printf("[");
            }
        }

        int idx = tensor_compute_idx(t, sh_iter.idx);

        printf("%f, ", t->data[idx]);

        for (int i = 0; i < t->sh.ndim; i++) {
            if (sh_iter.idx[i] == t->sh.dim[i] - 1) {
                printf("]");
            }
        }
    }
}

Tensor *tensor_add(Tensor *t1, Tensor *t2) {
    Shape sh;
    shape_copy(&t1->sh, &sh);

    Tensor *t = tensor_create(&sh, OP_ADD);

    Shape_Iter sh_iter = shape_iter_create(&sh);
    while (shape_iter_next(&sh_iter)) {
        int idx1 = tensor_compute_idx(t1, sh_iter.idx);
        int idx2 = tensor_compute_idx(t2, sh_iter.idx);
        int idx3 = tensor_compute_idx(t, sh_iter.idx);

        t->data[idx3] = t1->data[idx1] + t2->data[idx2];
    }

    return t;
}


void shape_print(Shape *sh) {
	printf("[");
	for (int i = 0; i < sh->ndim - 1; i++) {
		printf("%d, ", sh->dim[i]);
	}
	printf("%d]\n", sh->dim[sh->ndim - 1]);
}

int main() {
	srand(1);

	Tensor *t1 = tensor_arange(0, 5);
	Tensor *t2 = tensor_arange(1, 6);
	Tensor *t3 = tensor_add(t1, t2);

    /*

	Tensor *t2 = tensor_rand(2, shape);
	Tensor *t3 = tensor_add(t3, t4);

	tensor_print_shape(t3);
	tensor_print_stride(t3);
    */
	tensor_print(t3);

	return EXIT_SUCCESS;
}
