package core

import (
	"fmt"
)

// Shape is the dimension sizes of a tensor, e.g. [2, 3, 4].
type Shape []int

// Strides are byte offsets per axis (row-major).
type Strides []int

// ContiguousStrides computes row-major strides for a shape.
// Last axis stride = elemSize; strides[i] = strides[i+1] * shape[i+1].
func ContiguousStrides(shape Shape, elemSize uintptr) Strides {
	if len(shape) == 0 {
		return nil
	}
	strides := make(Strides, len(shape))
	strides[len(shape)-1] = int(elemSize)
	for i := len(shape) - 2; i >= 0; i-- {
		strides[i] = strides[i+1] * shape[i+1]
	}
	return strides
}

// NumElements returns the total number of elements (product of dimensions).
func (s Shape) NumElements() int {
	if len(s) == 0 {
		return 0
	}
	n := 1
	for _, d := range s {
		if d <= 0 {
			return 0
		}
		n *= d
	}
	return n
}

// BroadcastShapes applies NumPy-style broadcasting: pad shorter with 1s on the left,
// then compare right-to-left; equal dims stay, one is 1 → expand to other, else error.
func BroadcastShapes(a, b Shape) (Shape, error) {
	na, nb := len(a), len(b)
	maxLen := na
	if nb > maxLen {
		maxLen = nb
	}
	out := make(Shape, maxLen)
	// Pad left with 1s conceptually: out[i] for i < (maxLen - len) is 1
	for i := 0; i < maxLen; i++ {
		da, db := 1, 1
		if i >= maxLen-na {
			da = a[i-(maxLen-na)]
		}
		if i >= maxLen-nb {
			db = b[i-(maxLen-nb)]
		}
		if da == db {
			out[i] = da
		} else if da == 1 {
			out[i] = db
		} else if db == 1 {
			out[i] = da
		} else {
			return nil, fmt.Errorf("broadcast: incompatible shapes %v and %v", a, b)
		}
	}
	return out, nil
}
