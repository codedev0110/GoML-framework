package tensor

import "github.com/djeday123/goml/core"

// Re-export core types so other packages can use tensor.Shape, tensor.DType, etc.
// without importing core directly when using the tensor package.

type (
	// Shape is core.Shape.
	Shape = core.Shape
	// Strides is core.Strides.
	Strides = core.Strides
	// DType is core.DType.
	DType = core.DType
)

const (
	Float16   = core.Float16
	Float32   = core.Float32
	Float64   = core.Float64
	BFloat16  = core.BFloat16
	Int8      = core.Int8
	Int16     = core.Int16
	Int32     = core.Int32
	Int64     = core.Int64
)
