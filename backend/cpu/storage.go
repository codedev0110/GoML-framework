package cpu

import (
	"unsafe"

	"github.com/djeday123/goml/backend"
)

// storage wraps a byte slice for CPU memory.
type storage struct {
	buf []byte
	dev backend.Device
}

// NewStorage creates CPU storage from an existing byte slice (caller keeps ownership).
func NewStorage(buf []byte) backend.Storage {
	return &storage{buf: buf, dev: backend.CPU0}
}

// Alloc creates new CPU storage of the given byte length.
func Alloc(byteLen int) backend.Storage {
	return &storage{buf: make([]byte, byteLen), dev: backend.CPU0}
}

func (s *storage) Device() backend.Device { return s.dev }
func (s *storage) ByteLen() int           { return len(s.buf) }
func (s *storage) Bytes() []byte          { return s.buf }

func (s *storage) Ptr() uintptr {
	if len(s.buf) == 0 {
		return 0
	}
	return uintptr(unsafe.Pointer(&s.buf[0]))
}

func (s *storage) Free() {
	s.buf = nil
}
