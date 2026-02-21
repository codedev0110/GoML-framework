package core

import "unsafe"

// DType represents a tensor element type.
type DType uint8

const (
	Float16 DType = iota
	Float32
	Float64
	BFloat16
	Int8
	Int16
	Int32
	Int64
)

// Size returns the byte size of one element of this type.
func (d DType) Size() uintptr {
	switch d {
	case Float16, BFloat16:
		return 2
	case Float32, Int32:
		return 4
	case Float64, Int64:
		return 8
	case Int8:
		return 1
	case Int16:
		return 2
	default:
		return 4 // fallback
	}
}

// String returns a human-readable name for the type.
func (d DType) String() string {
	switch d {
	case Float16:
		return "float16"
	case Float32:
		return "float32"
	case Float64:
		return "float64"
	case BFloat16:
		return "bfloat16"
	case Int8:
		return "int8"
	case Int16:
		return "int16"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	default:
		return "unknown"
	}
}

// BFloat16Value is the storage type for BFloat16 (upper 16 bits of float32).
type BFloat16Value uint16

// Float32ToBFloat16 converts float32 to BFloat16 (reinterpret bits, right-shift 16).
func Float32ToBFloat16(f float32) BFloat16Value {
	u := *(*uint32)(unsafe.Pointer(&f))
	return BFloat16Value(u >> 16)
}

// Float32 converts BFloat16 to float32 (left-shift 16, reinterpret).
func (b BFloat16Value) Float32() float32 {
	u := uint32(b) << 16
	return *(*float32)(unsafe.Pointer(&u))
}
