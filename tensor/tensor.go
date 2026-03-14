package tensor

import (
	"fmt"
	"unsafe"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
)

// GradFn is the callback executed during backward to accumulate gradients.
type GradFn func()

// Tensor is the core multi‑dimensional array. The fields are unexported so
// that downstream packages must use the helper methods defined below.
//
// fields:
//   storage   : underlying buffer (device specific)
//   shape     : logical dimensions
//   strides   : byte offset between elements along each axis
//   dtype     : data type
//   offset    : byte offset from storage start (for slices/views)
//
//   requiresGrad : whether gradients should be tracked
//   grad         : accumulated gradient tensor
//   gradFn       : gradient function produced by the creating operation
//   isLeaf       : whether the tensor was created by user code
//
// The distinction between leaf and non‑leaf is important for optimizers; only
// leaves are updated by parameter updates in order to avoid mutating
// intermediate values.
//
// New tensors produced by operations are marked `isLeaf=false` and may carry a
// non‑nil gradFn.  Users constructing their own data (weights, inputs) should
// see `isLeaf=true`.

type Tensor struct {
	storage backend.Storage
	shape   core.Shape
	strides core.Strides
	dtype   core.DType
	offset  int // byte offset in storage for views

	requiresGrad bool
	grad         *Tensor
	gradFn       GradFn
	isLeaf       bool
}

// NewTensor constructs a tensor from existing storage.  The provided storage
// is *not* copied; callers are responsible for ensuring the buffer lives long
// enough.  Contiguous strides based on the dtype are computed automatically.
// The resulting tensor is marked as a leaf (isLeaf=true).  If callers are
// creating an intermediate value they should call SetLeaf(false) after
// construction.
func NewTensor(storage backend.Storage, shape core.Shape, dtype core.DType) *Tensor {
	strides := core.ContiguousStrides(shape, dtype.Size())
	return &Tensor{
		storage: storage,
		shape:   shape,
		strides: strides,
		dtype:   dtype,
		offset:  0,
		isLeaf:  true,
	}
}

// FromSlice creates a tensor on CPU from a Go slice.  Supported element types
// are []float32 and []int64; the slice is copied.  The shape must have the same
// number of elements as the slice.
func FromSlice(data interface{}, shape core.Shape) (*Tensor, error) {
	switch v := data.(type) {
	case []float32:
		return fromFloat32(v, shape)
	case []int64:
		return fromInt64(v, shape)
	default:
		return nil, fmt.Errorf("unsupported slice type %T", data)
	}
}

func fromFloat32(data []float32, shape core.Shape) (*Tensor, error) {
	if shape.NumElements() != len(data) {
		return nil, fmt.Errorf("shape %v has %d elements, data has %d", shape, shape.NumElements(), len(data))
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
	return NewTensor(storage, shape, core.Float32), nil
}

func fromInt64(data []int64, shape core.Shape) (*Tensor, error) {
	if shape.NumElements() != len(data) {
		return nil, fmt.Errorf("shape %v has %d elements, data has %d", shape, shape.NumElements(), len(data))
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
	return NewTensor(storage, shape, core.Int64), nil
}

// Convenience wrappers retained for backward compatibility.
func FromFloat32(data []float32, shape ...int) (*Tensor, error) {
	s := core.Shape(shape)
	return FromSlice(data, s)
}

func FromInt64(data []int64, shape ...int) (*Tensor, error) {
	s := core.Shape(shape)
	return FromSlice(data, s)
}

// Zeros returns a tensor filled with 0 on the specified device.
func Zeros(shape core.Shape, dtype core.DType, device backend.Device) (*Tensor, error) {
	be, err := backend.GetForDevice(device)
	if err != nil {
		return nil, err
	}
	n := shape.NumElements()
	byteLen := n * int(dtype.Size())
	storage, err := be.Alloc(byteLen)
	if err != nil {
		return nil, err
	}
	be.Fill(storage, n, 0)
	return NewTensor(storage, shape, dtype), nil
}

// Ones returns a tensor filled with 1 on the specified device.
func Ones(shape core.Shape, dtype core.DType, device backend.Device) (*Tensor, error) {
	out, err := Zeros(shape, dtype, device)
	if err != nil {
		return nil, err
	}
	// fill with ones (only float32 supported for now)
	if dtype == core.Float32 {
		buf := out.ToFloat32Slice()
		for i := range buf {
			buf[i] = 1
		}
	}
	return out, nil
}

// Arange creates a 1‑D tensor containing [start, start+1, ..., stop-1].
func Arange(start, stop int, dtype core.DType, device backend.Device) (*Tensor, error) {
	n := stop - start
	if n <= 0 {
		return nil, fmt.Errorf("invalid arange %d..%d", start, stop)
	}
	be, err := backend.GetForDevice(device)
	if err != nil {
		return nil, err
	}
	byteLen := n * int(dtype.Size())
	storage, err := be.Alloc(byteLen)
	if err != nil {
		return nil, err
	}
	// only implement for float32 and int64
	if dtype == core.Float32 {
		tmp := Float32FromBytes(storage.Bytes())
		for i := 0; i < n; i++ {
			tmp[i] = float32(start + i)
		}
	} else if dtype == core.Int64 {
		tmp := Int64FromBytes(storage.Bytes())
		for i := 0; i < n; i++ {
			tmp[i] = int64(start + i)
		}
	}
	return NewTensor(storage, core.Shape{n}, dtype), nil
}

// NumElements returns the total number of elements.
func (t *Tensor) NumElements() int {
	return t.shape.NumElements()
}

// NDim returns the number of dimensions.
func (t *Tensor) NDim() int {
	return len(t.shape)
}

// IsContiguous returns true if the tensor uses standard row-major strides.
func (t *Tensor) IsContiguous() bool {
	expected := core.ContiguousStrides(t.shape, t.dtype.Size())
	if len(expected) != len(t.strides) {
		return false
	}
	for i := range expected {
		if expected[i] != t.strides[i] {
			return false
		}
	}
	return true
}

// View returns a zero‑copy reshape of t.  The underlying storage is shared.
// The tensor must be contiguous and the element counts must agree.
func (t *Tensor) View(newShape core.Shape) (*Tensor, error) {
	if !t.IsContiguous() {
		return nil, fmt.Errorf("cannot view non-contiguous tensor")
	}
	if newShape.NumElements() != t.NumElements() {
		return nil, fmt.Errorf("view shape %v has %d elements, tensor has %d", newShape, newShape.NumElements(), t.NumElements())
	}
	return &Tensor{
		storage:      t.storage,
		shape:        newShape,
		strides:      core.ContiguousStrides(newShape, t.dtype.Size()),
		dtype:        t.dtype,
		offset:       t.offset,
		requiresGrad: t.requiresGrad,
		isLeaf:       false,
	}, nil
}

// Transpose swaps two axes in the tensor (zero-copy).  The returned tensor
// shares storage; only shape and strides are modified.
func (t *Tensor) Transpose(i, j int) (*Tensor, error) {
	n := t.NDim()
	if i < 0 || i >= n || j < 0 || j >= n {
		return nil, fmt.Errorf("transpose axes out of range")
	}
	newShape := make(core.Shape, n)
	copy(newShape, t.shape)
	newShape[i], newShape[j] = newShape[j], newShape[i]
	newStrides := make(core.Strides, n)
	copy(newStrides, t.strides)
	newStrides[i], newStrides[j] = newStrides[j], newStrides[i]
	return &Tensor{
		storage:      t.storage,
		shape:        newShape,
		strides:      newStrides,
		dtype:        t.dtype,
		offset:       t.offset,
		requiresGrad: t.requiresGrad,
		isLeaf:       false,
	}, nil
}

// T is a convenience that transposes the last two dimensions.
func (t *Tensor) T() (*Tensor, error) {
	if t.NDim() < 2 {
		return nil, fmt.Errorf("T() requires at least 2 dimensions")
	}
	return t.Transpose(t.NDim()-2, t.NDim()-1)
}

// Accessors --------------------------------------------------------------
func (t *Tensor) Storage() backend.Storage { return t.storage }
func (t *Tensor) Shape() core.Shape        { return t.shape }
func (t *Tensor) Strides() core.Strides    { return t.strides }
func (t *Tensor) DType() core.DType        { return t.dtype }
func (t *Tensor) Offset() int              { return t.offset }
func (t *Tensor) RequiresGrad() bool       { return t.requiresGrad }
func (t *Tensor) Grad() *Tensor            { return t.grad }
func (t *Tensor) GradFn() GradFn           { return t.gradFn }
func (t *Tensor) IsLeaf() bool             { return t.isLeaf }

// Mutators ---------------------------------------------------------------
func (t *Tensor) SetRequiresGrad(v bool) { t.requiresGrad = v }
func (t *Tensor) SetGrad(g *Tensor)      { t.grad = g }
func (t *Tensor) SetGradFn(fn GradFn)    { t.gradFn = fn }
func (t *Tensor) SetLeaf(v bool)         { t.isLeaf = v }

// Memory helpers ---------------------------------------------------------

// ToFloat32Slice returns a float32 slice view of the tensor data.  Panics if
// the dtype is not Float32 or if the storage is on a non‑CPU backend.
func (t *Tensor) ToFloat32Slice() []float32 {
	if t.dtype != core.Float32 {
		panic("ToFloat32Slice() only valid for Float32 tensor")
	}
	return Float32FromBytes(t.storage.Bytes())
}

// ToInt64Slice returns an int64 slice view of the tensor data.
func (t *Tensor) ToInt64Slice() []int64 {
	if t.dtype != core.Int64 {
		panic("ToInt64Slice() only valid for Int64 tensor")
	}
	return Int64FromBytes(t.storage.Bytes())
}

// Clone allocates a new tensor with the same shape and copies data.
// The clone is a leaf and has no gradient information.
func (t *Tensor) Clone() (*Tensor, error) {
	be, err := backend.GetForDevice(t.storage.Device())
	if err != nil {
		return nil, err
	}
	byteLen := t.NumElements() * int(t.dtype.Size())
	newStorage, err := be.Alloc(byteLen)
	if err != nil {
		return nil, err
	}
	if err := be.Copy(newStorage, t.storage, byteLen); err != nil {
		newStorage.Free()
		return nil, err
	}
	return NewTensor(newStorage, t.shape, t.dtype), nil
}
