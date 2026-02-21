package tensor

import (
	"unsafe"

	"github.com/djeday123/goml/core"
)

// Float32FromBytes returns a float32 slice that shares memory with b.
// Caller must ensure b is not modified after use and has length divisible by 4.
func Float32FromBytes(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// BytesFromFloat32 returns a byte slice that shares memory with f.
func BytesFromFloat32(f []float32) []byte {
	if len(f) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&f[0])), len(f)*4)
}

// Int64FromBytes returns an int64 slice that shares memory with b.
func Int64FromBytes(b []byte) []int64 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*int64)(unsafe.Pointer(&b[0])), len(b)/8)
}

// UintptrFromStorage returns the raw pointer and byte length for a storage.
// Used to bridge tensor and backend.
func UintptrFromStorage(storage interface{ Ptr() uintptr; ByteLen() int }) (uintptr, int) {
	return storage.Ptr(), storage.ByteLen()
}

// ElemSize returns the byte size of one element for the given DType.
func ElemSize(d core.DType) uintptr {
	return d.Size()
}
