package core

import (
	"testing"
)

func TestContiguousStrides(t *testing.T) {
	shape := Shape{2, 3, 4}
	strides := ContiguousStrides(shape, 4)
	if len(strides) != 3 || strides[0] != 48 || strides[1] != 16 || strides[2] != 4 {
		t.Fatalf("ContiguousStrides([2,3,4], 4) = %v, want [48, 16, 4]", strides)
	}
}

func TestBroadcastShapes(t *testing.T) {
	a, b := Shape{2, 3}, Shape{1, 3}
	out, err := BroadcastShapes(a, b)
	if err != nil || len(out) != 2 || out[0] != 2 || out[1] != 3 {
		t.Fatalf("BroadcastShapes([2,3], [1,3]) = %v, %v", out, err)
	}
}
