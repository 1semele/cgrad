#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <stdarg.h>

#define MAX_DIMS 8

#define MAX(_a, _b) ((_a) > (_b) ? (_a) : (_b))

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
    int data_size;
} Tensor;

typedef struct {
	int idx[MAX_DIMS];
	Tensor *t;
    int full_idx;
} Tensor_Iter;

void shape_print(Shape *sh) {
	printf("[");
	for (int i = 0; i < sh->ndim - 1; i++) {
		printf("%d, ", sh->dim[i]);
	}
	printf("%d]\n", sh->dim[sh->ndim - 1]);
}

bool shape_equal(Shape *sh1, Shape *sh2) {
    if (sh1->ndim != sh2->ndim) {
        return false;
    }

    for (int i = 0; i < sh1->ndim; i++) {
        if (sh1->dim[i] != sh2->dim[i]) {
            return false;
        }
    }

    return true;
}

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

Tensor *tensor_create_with_size(Shape *sh, Op op, int space_needed) {
	Tensor *t = malloc(sizeof(Tensor));

    /* [sh] can optionally be NULL, then we initialize NOTHING about its dim, stride, etc. */
    if (sh) {
        shape_copy(sh, &t->sh);
        int stride = 1;
        for (int i = sh->ndim - 1; i >= 0; i--) {
            t->stride[i] = stride;
            stride *= sh->dim[i];
        }
    }

	t->op = op;
	t->data = malloc(sizeof(float) * space_needed);
    t->data_size = space_needed;

	return t;
}

Tensor *tensor_create(Shape *sh, Op op) {
	return tensor_create_with_size(sh, op, shape_space_needed(sh));
}


void tensor_copy_data(Tensor *src, Tensor *dst) {
    if (src->data_size != dst->data_size) {
        printf("Error copying data: mismatched sizes\n");
        exit(EXIT_FAILURE);
    }

    memcpy(dst->data, src->data, sizeof(float) * src->data_size);
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

void tensor_print_stride(Tensor *t) {
    printf("[");
    for (int i = 0; i < t->sh.ndim; i++) {
        printf("%d, ", t->stride[i]);
    }
    printf("]\n");
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
    printf("\n");
}

void tensor_broadcast(Tensor *t1, Tensor *t2, Tensor **new_t1_out, Tensor **new_t2_out) {
    Shape *sh1 = &t1->sh;
    Shape *sh2 = &t2->sh;

    int max_dim = MAX(sh1->ndim, sh2->ndim);

    Tensor *new_t1 = tensor_create_with_size(NULL, OP_NONE, shape_space_needed(&t1->sh));
    Tensor *new_t2 = tensor_create_with_size(NULL, OP_NONE, shape_space_needed(&t2->sh));
    tensor_copy_data(t1, new_t1);
    tensor_copy_data(t2, new_t2);

    Shape sh = {
        .ndim = max_dim,
    };

    for (int i = max_dim - 1; i >= 0; i--) {
        int dim1_idx = i + max_dim - sh1->ndim;
        int dim2_idx = i + max_dim - sh2->ndim;
        int dim1, dim2;

        if (dim1_idx < 0) {
            dim1 = 1;
        } else {
            dim1 = sh1->dim[dim1_idx];
        }

        if (dim2_idx < 0) {
            dim2 = 1;
        } else {
            dim2 = sh2->dim[dim2_idx];
        }

        if (dim1 == dim2) {
            // Nothing is required! Strides should all be in place.
            new_t1->stride[i] = t1->stride[i];
            new_t2->stride[i] = t2->stride[i];
        } if (dim1 == 1) {
            new_t1->stride[i] = 0;
            new_t2->stride[i] = t2->stride[i];
        } if (dim2 == 1)  {
            new_t1->stride[i] = t1->stride[i];
            new_t2->stride[i] = 0;
        } else {
            printf("Broadcasting error\n");
            shape_print(sh1);
            shape_print(sh2);
            exit(EXIT_FAILURE);
        }

        sh.dim[i] = MAX(dim1, dim2);
    }

    shape_copy(&sh, &new_t1->sh);
    shape_copy(&sh, &new_t2->sh);

    *new_t1_out = new_t1;
    *new_t2_out = new_t2;
}

Tensor *tensor_add(Tensor *t1, Tensor *t2) {
    if (!shape_equal(&t1->sh, &t2->sh)) {
        Tensor *new_t1, *new_t2;
        tensor_broadcast(t1, t2, &new_t1, &new_t2);
        t1 = new_t1;
        t2 = new_t2;
    }

    Tensor *t = tensor_create(&t1->sh, OP_ADD);

    Shape_Iter sh_iter = shape_iter_create(&t1->sh);
    while (shape_iter_next(&sh_iter)) {
        int idx1 = tensor_compute_idx(t1, sh_iter.idx);
        int idx2 = tensor_compute_idx(t2, sh_iter.idx);
        int idx3 = tensor_compute_idx(t, sh_iter.idx);

        t->data[idx3] = t1->data[idx1] + t2->data[idx2];
    }

    return t;
}

Tensor *tensor_mul(Tensor *t1, Tensor *t2) {
    if (!shape_equal(&t1->sh, &t2->sh)) {
        Tensor *new_t1, *new_t2;
        tensor_broadcast(t1, t2, &new_t1, &new_t2);
        t1 = new_t1;
        t2 = new_t2;
    }

    Tensor *t = tensor_create(&t1->sh, OP_ADD);

    Shape_Iter sh_iter = shape_iter_create(&t1->sh);
    while (shape_iter_next(&sh_iter)) {
        int idx1 = tensor_compute_idx(t1, sh_iter.idx);
        int idx2 = tensor_compute_idx(t2, sh_iter.idx);
        int idx3 = tensor_compute_idx(t, sh_iter.idx);

        t->data[idx3] = t1->data[idx1] * t2->data[idx2];
    }

    return t;
}

int main() {
	srand(1);

	Tensor *t1 = tensor_arange(0, 5);
	Tensor *t2 = tensor_arange(1, 2);
	Tensor *t3 = tensor_arange(5, 10);
	Tensor *t4 = tensor_add(t1, t2);
	Tensor *t5 = tensor_mul(t3, t4);

	tensor_print(t1);
	tensor_print(t2);
	tensor_print(t3);
	tensor_print(t4);
	tensor_print(t5);

	return EXIT_SUCCESS;
}
