package backend

import (
	"errors"
	"fmt"

	"github.com/djeday123/goml/core"
)

// DeviceType identifies the kind of hardware.
type DeviceType uint8

const (
	CPU DeviceType = iota
	CUDA
	ROCm
	Metal
	Vulkan
)

// Device identifies a specific device (e.g. GPU 0).
type Device struct {
	Type  DeviceType
	Index int
}

// CPU0 is the default CPU device.
var CPU0 = Device{Type: CPU, Index: 0}

// Storage represents raw memory on a device.
// Ptr() is the bridge to raw hardware (RAM address for CPU, device pointer for GPU).
type Storage interface {
	Device() Device
	Ptr() uintptr
	Bytes() []byte // CPU only; nil for GPU
	ByteLen() int
	Free()
}

// Backend is the contract every hardware backend must implement.
type Backend interface {
	Name() string
	DeviceType() DeviceType

	Alloc(byteLen int) (Storage, error)
	Free(s Storage)
	Copy(dst, src Storage, byteLen int) error
	ToDevice(dst Device, src Storage) (Storage, error)

	// Unary (dst, src, nElems; float32)
	Neg(dst, src Storage, nElems int) error
	Abs(dst, src Storage, nElems int) error
	Exp(dst, src Storage, nElems int) error
	Log(dst, src Storage, nElems int) error
	Sqrt(dst, src Storage, nElems int) error
	Tanh(dst, src Storage, nElems int) error
	Relu(dst, src Storage, nElems int) error
	Gelu(dst, src Storage, nElems int) error
	Sigmoid(dst, src Storage, nElems int) error
	Silu(dst, src Storage, nElems int) error

	// Binary with broadcasting: dst = a op b (shape = broadcast(aShape, bShape))
	Add(dst, a, b Storage, aShape, bShape core.Shape, aStrides, bStrides core.Strides, outShape core.Shape) error
	Sub(dst, a, b Storage, aShape, bShape core.Shape, aStrides, bStrides core.Strides, outShape core.Shape) error
	Mul(dst, a, b Storage, aShape, bShape core.Shape, aStrides, bStrides core.Strides, outShape core.Shape) error
	Div(dst, a, b Storage, aShape, bShape core.Shape, aStrides, bStrides core.Strides, outShape core.Shape) error

	// Reductions: axis -1 = all axes; keepDim = keep reduced dim as 1
	Sum(dst, src Storage, srcShape core.Shape, srcStrides core.Strides, axis int, keepDim bool) error
	Max(dst, src Storage, srcShape core.Shape, srcStrides core.Strides, axis int, keepDim bool) error
	Mean(dst, src Storage, srcShape core.Shape, srcStrides core.Strides, axis int, keepDim bool) error

	// MatMul: C = A @ B. A [..., M, K], B [..., K, N], C [..., M, N]. Batched by leading dims.
	MatMul(dst, a, b Storage, batchSize, M, N, K int) error

	// Softmax along last axis; shape = full shape of src.
	Softmax(dst, src Storage, shape core.Shape, strides core.Strides) error

	// LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta. Normalize over last axis.
	LayerNorm(dst, x, gamma, beta, mean, var_ Storage, shape core.Shape, strides core.Strides, eps float32) error

	// Embedding: indices (int64), table (float32), out same dtype as table.
	Embedding(dst, table Storage, indices Storage, tableRows, tableCols int) error

	// RoPE: apply rotary positional embedding to x in-place (or dst = RoPE(x)); last dim = head_dim.
	RoPE(dst, x Storage, shape core.Shape, strides core.Strides, base float64, startPos, seqLen int) error

	// ScaledDotProductAttention: Q,K,V [batch, heads, seq, head_dim]; causal mask optional.
	ScaledDotProductAttention(dst, q, k, v Storage, batch, heads, seq, headDim int, causal bool) error

	Fill(dst Storage, nElems int, value float32) error
	Arange(dst Storage, nElems int, start, step float32) error
	Where(dst, cond, a, b Storage, nElems int) error // cond nonzero -> a else b (float32)
}

var registry = make(map[DeviceType]Backend)

// Register adds a backend for its device type.
func Register(b Backend) {
	registry[b.DeviceType()] = b
}

// Get returns the backend for a device type.
func Get(dt DeviceType) (Backend, error) {
	b, ok := registry[dt]
	if !ok {
		return nil, fmt.Errorf("no backend registered for device type %v", dt)
	}
	return b, nil
}

// GetForDevice returns the backend that handles the given device.
func GetForDevice(d Device) (Backend, error) {
	return Get(d.Type)
}

// ErrUnsupported is returned when an operation is not supported.
var ErrUnsupported = errors.New("operation not supported")
