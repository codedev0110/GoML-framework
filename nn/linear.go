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

	// Store forward data for backward
	out.Backward = func() {
		l.BackwardFunction(x, out)
	}

	return out, nil
}

// BackwardFunction computes and accumulates gradients for weights and bias, and input.
// Expects out.Grad to be set (gradient with respect to output).
// dL/dW = dL/dOut^T @ x, dL/dBias = sum(dL/dOut), dL/dX = dL/dOut @ W
func (l *Linear) BackwardFunction(x, out *tensor.Tensor) {
	if out.Grad == nil {
		return
	}

	be, err := backend.GetForDevice(out.Storage.Device())
	if err != nil {
		return
	}

	batch := x.NumElements() / l.InSize

	// dL/dBias = sum(dL/dOut) over batch dimension (reduce axis 0)
	// out.Grad shape: [batch, OutSize], reduce to [OutSize]
	if l.Bias.Grad == nil {
		biasGradSize := l.OutSize * 4
		biasGradStorage, _ := be.Alloc(biasGradSize)
		be.Fill(biasGradStorage, l.OutSize, 0)
		l.Bias.Grad = tensor.New(biasGradStorage, core.Shape{l.OutSize}, core.ContiguousStrides(core.Shape{l.OutSize}, 4), core.Float32)
	}

	// Sum gradients over batch
	biasGradF := l.Bias.Grad.Float32()
	outGradF := out.Grad.Float32()
	for b := 0; b < batch; b++ {
		for o := 0; o < l.OutSize; o++ {
			biasGradF[o] += outGradF[b*l.OutSize+o]
		}
	}

	// dL/dW = dL/dOut^T @ x
	// out.Grad: [batch, OutSize], x: [batch, InSize]
	// dL/dW: [OutSize, InSize]
	if l.W.Grad == nil {
		wGradSize := l.OutSize * l.InSize * 4
		wGradStorage, _ := be.Alloc(wGradSize)
		be.Fill(wGradStorage, l.OutSize*l.InSize, 0)
		l.W.Grad = tensor.New(wGradStorage, core.Shape{l.OutSize, l.InSize},
			core.ContiguousStrides(core.Shape{l.OutSize, l.InSize}, 4), core.Float32)
	}

	// dW += dL/dOut^T @ x (manually compute since backend MatMul is transposed)
	outGradF = out.Grad.Float32()
	xF := x.Float32()
	wGradF := l.W.Grad.Float32()

	for o := 0; o < l.OutSize; o++ {
		for i := 0; i < l.InSize; i++ {
			grad := float32(0)
			for b := 0; b < batch; b++ {
				grad += outGradF[b*l.OutSize+o] * xF[b*l.InSize+i]
			}
			wGradF[o*l.InSize+i] += grad
		}
	}

	// dL/dX = dL/dOut @ W [batch, OutSize] @ [OutSize, InSize] = [batch, InSize]
	// Compute gradient for input x
	if x.Grad == nil {
		xGradSize := batch * l.InSize * 4
		xGradStorage, _ := be.Alloc(xGradSize)
		be.Fill(xGradStorage, batch*l.InSize, 0)
		x.Grad = tensor.New(xGradStorage, x.Shape, core.ContiguousStrides(x.Shape, 4), core.Float32)
	}

	// dX += dOut @ W
	xGradF := x.Float32()
	for b := 0; b < batch; b++ {
		for i := 0; i < l.InSize; i++ {
			grad := float32(0)
			for o := 0; o < l.OutSize; o++ {
				grad += outGradF[b*l.OutSize+o] * wGradF[o*l.InSize+i]
			}
			xGradF[b*l.InSize+i] += grad
		}
	}

	// Recursively call backward on input if it has a backward function
	if x.Backward != nil {
		x.Backward()
	}
}
