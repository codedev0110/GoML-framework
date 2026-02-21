package nn

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// Linear is y = x @ W^T + bias. InSize, OutSize; W is [OutSize, InSize], bias [OutSize].
type Linear struct {
	W      *tensor.Tensor // [OutSize, InSize]
	Bias   *tensor.Tensor // [OutSize]
	InSize int
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
	// x [batch..., InSize], W [OutSize, InSize]. x @ W^T = [batch..., OutSize].
	// So we need matmul(x, W^T). W^T is [InSize, OutSize]. So matmul(x, W^T): x [..., InSize], W^T [InSize, OutSize].
	// We have W [OutSize, InSize]. So we do x @ W^T by doing matmul(x, W) with W used as "W^T" i.e. we need
	// output[i,j] = sum_k x[i,k] * W^T[k,j] = sum_k x[i,k] * W[j,k]. So (x @ W^T) = matmul(x, W^T).
	// Backend MatMul: A [..., M, K], B [..., K, N]. So we want A=x [batch, inSize], B=W^T [inSize, outSize].
	// So we need to pass W transposed. W is [OutSize, InSize]; W^T is [InSize, OutSize].
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
	// x [batch, InSize], W^T [InSize, OutSize] -> out [batch, OutSize]. batchSize=1.
	be.MatMul(outStorage, x.Storage, wt.Storage, 1, batch, l.OutSize, l.InSize)
	// Add bias: broadcast [OutSize] to [batch, OutSize].
	biasStorage := l.Bias.Storage
	biasShape := core.Shape{l.OutSize}
	outShape := core.Shape{batch, l.OutSize}
	biasStrides := core.ContiguousStrides(biasShape, 4)
	outStrides := core.ContiguousStrides(outShape, 4)
	addStorage, _ := be.Alloc(outSize * 4)
	be.Add(addStorage, outStorage, biasStorage, outShape, biasShape, outStrides, biasStrides, outShape)
	be.Free(outStorage)
	out := tensor.New(addStorage, outShape, outStrides, core.Float32)
	return out, nil
}
