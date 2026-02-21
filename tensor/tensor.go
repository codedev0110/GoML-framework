package tensor

import (
	"fmt"
	"unsafe"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

// Tensor is the core multi-dimensional array: storage + shape + strides + dtype.
// Grad is set during backward; Backward is the gradient callback for autograd.
type Tensor struct {
	Storage  backend.Storage
	Shape    core.Shape
	Strides  core.Strides
	DType    core.DType
	Grad     *Tensor   // accumulated gradient (optional)
	Backward func()    // called during backward pass (optional)
	RequiresGrad bool
}

// New creates a tensor from existing storage, shape, and strides.
// If strides is nil, contiguous row-major strides are computed.
func New(storage backend.Storage, shape core.Shape, strides core.Strides, dtype core.DType) *Tensor {
	if strides == nil {
		strides = core.ContiguousStrides(shape, dtype.Size())
	}
	return &Tensor{
		Storage: storage,
		Shape:   shape,
		Strides: strides,
		DType:   dtype,
	}
}

// NumElements returns the total number of elements.
func (t *Tensor) NumElements() int {
	return t.Shape.NumElements()
}

// Contiguous returns true if the tensor is row-major contiguous.
func (t *Tensor) Contiguous() bool {
	expected := core.ContiguousStrides(t.Shape, t.DType.Size())
	if len(expected) != len(t.Strides) {
		return false
	}
	for i := range expected {
		if expected[i] != t.Strides[i] {
			return false
		}
	}
	return true
}

// View returns a new tensor sharing storage with t but with the given shape.
// The product of shape must equal t.NumElements(). Strides are recomputed as contiguous.
func (t *Tensor) View(shape ...int) (*Tensor, error) {
	s := core.Shape(shape)
	if s.NumElements() != t.NumElements() {
		return nil, fmt.Errorf("view shape %v has %d elements, tensor has %d", shape, s.NumElements(), t.NumElements())
	}
	strides := core.ContiguousStrides(s, t.DType.Size())
	return New(t.Storage, s, strides, t.DType), nil
}

// Transpose returns a new tensor with axes swapped. Only 2D for simplicity.
func (t *Tensor) Transpose() (*Tensor, error) {
	if len(t.Shape) != 2 {
		return nil, fmt.Errorf("transpose only supported for 2D tensors, got shape %v", t.Shape)
	}
	// new shape [N, M], new strides: row stride = old col stride, col stride = old row stride
	newShape := core.Shape{t.Shape[1], t.Shape[0]}
	newStrides := core.Strides{t.Strides[1], t.Strides[0]}
	return New(t.Storage, newShape, newStrides, t.DType), nil
}

// FromFloat32 creates a new CPU tensor from a float32 slice (copy; contiguous).
func FromFloat32(data []float32, shape ...int) (*Tensor, error) {
	s := core.Shape(shape)
	if s.NumElements() != len(data) {
		return nil, fmt.Errorf("shape %v has %d elements, data has %d", shape, s.NumElements(), len(data))
	}
	be, err := backend.GetForDevice(backend.CPU0)
	if err != nil {
		return nil, err
	}
	byteLen := len(data) * 4
	storage, err := be.Alloc(byteLen)
	if err != nil {
		return nil, err
	}
	copy(storage.Bytes(), BytesFromFloat32(data))
	strides := core.ContiguousStrides(s, core.Float32.Size())
	return New(storage, s, strides, core.Float32), nil
}

// FromInt64 creates a new CPU tensor from an int64 slice (e.g. token indices).
func FromInt64(data []int64, shape ...int) (*Tensor, error) {
	s := core.Shape(shape)
	if s.NumElements() != len(data) {
		return nil, fmt.Errorf("shape %v has %d elements, data has %d", shape, s.NumElements(), len(data))
	}
	be, err := backend.GetForDevice(backend.CPU0)
	if err != nil {
		return nil, err
	}
	byteLen := len(data) * 8
	storage, err := be.Alloc(byteLen)
	if err != nil {
		return nil, err
	}
	b := storage.Bytes()
	dst := unsafe.Slice((*int64)(unsafe.Pointer(&b[0])), len(data))
	copy(dst, data)
	strides := core.ContiguousStrides(s, core.Int64.Size())
	return New(storage, s, strides, core.Int64), nil
}

// Float32 returns the underlying float32 slice for CPU tensors (shared memory).
// Panics if not CPU or not Float32 dtype.
func (t *Tensor) Float32() []float32 {
	if t.DType != core.Float32 {
		panic("Float32() only for Float32 tensors")
	}
	return Float32FromBytes(t.Storage.Bytes())
}

// Int64 returns the underlying int64 slice for CPU tensors.
func (t *Tensor) Int64() []int64 {
	if t.DType != core.Int64 {
		panic("Int64() only for Int64 tensors")
	}
	return Int64FromBytes(t.Storage.Bytes())
}

// Clone allocates a new tensor with the same shape and copies data.
func (t *Tensor) Clone() (*Tensor, error) {
	be, err := backend.GetForDevice(t.Storage.Device())
	if err != nil {
		return nil, err
	}
	byteLen := t.NumElements() * int(t.DType.Size())
	newStorage, err := be.Alloc(byteLen)
	if err != nil {
		return nil, err
	}
	if err := be.Copy(newStorage, t.Storage, byteLen); err != nil {
		newStorage.Free()
		return nil, err
	}
	return New(newStorage, t.Shape, core.ContiguousStrides(t.Shape, t.DType.Size()), t.DType), nil
}
