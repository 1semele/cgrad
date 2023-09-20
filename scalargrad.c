#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef enum {
	OP_CONST,
	OP_ADD,
	OP_MUL,
	OP_SIN,
	OP_LOG,
	OP_EXP,
} Value_Op;

typedef struct Value {
	Value_Op op;
	float val;
	struct Value *p1, *p2;
	float grad;
} Value;

const char *op_names[] = {
	[OP_CONST] = "const",
	[OP_ADD] = "add",
	[OP_MUL] = "mul",
	[OP_SIN] = "sin",
	[OP_LOG] = "log",
	[OP_EXP] = "exp",
};

static Value *value_create(Value_Op op, float start_val) {
	Value *val = malloc(sizeof(Value));
	val->op = op;
	val->val = start_val;
	val->p1 = val->p2 = NULL;
	return val;
}

static Value *value_const(float val) {
	return value_create(OP_CONST, val);
}

static Value *value_add(Value *v1, Value *v2) {
	Value *val = value_create(OP_ADD, v1->val + v2->val);
	val->p1 = v1;
	val->p2 = v2;
	return val;
}

static Value *value_mul(Value *v1, Value *v2) {
	Value *val = value_create(OP_MUL, v1->val * v2->val);
	val->p1 = v1;
	val->p2 = v2;
	return val;
}

static Value *value_sub(Value *v1, Value *v2) {
	Value *neg = value_const(-1.0);
	Value *v3 = value_mul(v2, neg);
	return value_add(v1, v3);
}

static Value *value_sin(Value *v) {
	Value *val = value_create(OP_SIN, sin(v->val));
	val->p1 = v;
	return val;
}

static Value *value_log(Value *v) {
	Value *val = value_create(OP_LOG, log(v->val));
	val->p1 = v;
	return val;
}

static Value *value_exp(Value *v) {
	Value *val = value_create(OP_EXP, exp(v->val));
	val->p1 = v;
	return val;
}


static void value_backward_rec(Value *v) {
	switch (v->op) {
		case OP_ADD:
			v->p1->grad += v->grad;
			v->p2->grad += v->grad;
			value_backward_rec(v->p1);
			value_backward_rec(v->p2);
			break;
		case OP_MUL:
			v->p1->grad += v->grad * v->p2->val;
			v->p2->grad += v->grad * v->p1->val;
			value_backward_rec(v->p1);
			value_backward_rec(v->p2);
			break;
		case OP_SIN:
			v->p1->grad += v->grad * cos(v->p1->val);
			value_backward_rec(v->p1);
			break;
		case OP_LOG:
			v->p1->grad += v->grad / v->p1->val;
			value_backward_rec(v->p1);
			break;
		case OP_EXP:
			v->p1->grad += v->grad / v->p1->val;
			value_backward_rec(v->p1);
			break;
		default:
			break;
	}
}

static void value_backward(Value *v) {
	v->grad = 1.0f;
	value_backward_rec(v);
}

static void value_print(Value *val) {
	printf("%s: %f %f\n", op_names[val->op], val->val, val->grad);
}

int main() {
	Value *x1 = value_const(2);
	Value *x2 = value_const(5);

	Value *log_exp = value_log(x1);
	Value *mul_exp = value_mul(x1, x2);
	Value *sin_exp = value_sin(x2);
	
	Value *v1 = value_add(log_exp, mul_exp);
	Value *v2 = value_sub(v1, sin_exp);

	value_print(v2);
	value_backward(v2);
	
	value_print(x1);
	
	return EXIT_SUCCESS;
}
