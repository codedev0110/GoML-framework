package nn

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// Linear is y = x @ W^T + bias. InSize, OutSize; W is [OutSize, InSize], bias [OutSize].
type Linear struct {
	W       *tensor.Tensor // [OutSize, InSize]
	Bias    *tensor.Tensor // [OutSize]
	InSize  int
	OutSize int
}

// NewLinear creates a linear layer with W and bias (caller provides initialized tensors).
func NewLinear(inSize, outSize int, W, bias *tensor.Tensor) (*Linear, error) {
	if W.NumElements() != outSize*inSize || bias.NumElements() != outSize {
		return nil, fmt.Errorf("Linear: W must be [%d,%d], bias [%d]", outSize, inSize, outSize)
	}
	return &Linear{W: W, Bias: bias, InSize: inSize, OutSize: outSize}, nil
}

// Forward computes x @ W^T + bias. x: [..., InSize], out: [..., OutSize].
func (l *Linear) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	wt, err := l.W.Transpose()
	if err != nil {
		return nil, err
	}
	// x: [..., InSize] -> treat as [batch, InSize] with batch = product of leading dims.
	batch := x.NumElements() / l.InSize
	be, err := backend.GetForDevice(x.Storage.Device())
	if err != nil {
		return nil, err
	}
	outSize := batch * l.OutSize
	outStorage, err := be.Alloc(outSize * 4)
	if err != nil {
		return nil, err
	}
	be.Fill(outStorage, outSize, 0)
	be.MatMul(outStorage, x.Storage, wt.Storage, 1, batch, l.OutSize, l.InSize)
	// Add bias: broadcast [OutSize] to [batch, OutSize].
	biasStorage := l.Bias.Storage
	biasShape := core.Shape{l.OutSize}
	flatOutShape := core.Shape{batch, l.OutSize}
	biasStrides := core.ContiguousStrides(biasShape, 4)
	flatOutStrides := core.ContiguousStrides(flatOutShape, 4)
	addStorage, _ := be.Alloc(outSize * 4)
	be.Add(addStorage, outStorage, biasStorage, flatOutShape, biasShape, flatOutStrides, biasStrides, flatOutShape)
	be.Free(outStorage)

	// Restore original leading dimensions: if input was [B,S,D], output is [B,S,OutSize]
	finalShape := make(core.Shape, len(x.Shape))
	copy(finalShape, x.Shape)
	finalShape[len(finalShape)-1] = l.OutSize
	finalStrides := core.ContiguousStrides(finalShape, 4)

	out := tensor.New(addStorage, finalShape, finalStrides, core.Float32)
	return out, nil
}
