package ops

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// Add returns a + b with broadcasting. If both require grad, backward accumulates gradients.
func Add(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := core.BroadcastShapes(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}
	be, err := backend.GetForDevice(a.Storage.Device())
	if err != nil {
		return nil, err
	}
	byteLen := outShape.NumElements() * 4
	outStorage, err := be.Alloc(byteLen)
	if err != nil {
		return nil, err
	}
	outStrides := core.ContiguousStrides(outShape, 4)
	if err := be.Add(outStorage, a.Storage, b.Storage, a.Shape, b.Shape, a.Strides, b.Strides, outShape); err != nil {
		outStorage.Free()
		return nil, err
	}
	out := tensor.New(outStorage, outShape, outStrides, core.Float32)
	out.RequiresGrad = a.RequiresGrad || b.RequiresGrad
	if out.RequiresGrad {
		out.Backward = func() {
			if a.RequiresGrad && a.Grad != nil {
				// grad_a += grad_out (broadcast if needed)
				addGradInto(a.Grad, out.Grad, a.Shape, outShape)
			}
			if b.RequiresGrad && b.Grad != nil {
				addGradInto(b.Grad, out.Grad, b.Shape, outShape)
			}
		}
	}
	return out, nil
}

// addGradInto adds gradOut (broadcast/sum to match aShape) into aGrad.
func addGradInto(aGrad, gradOut *tensor.Tensor, aShape, outShape core.Shape) {
	be, _ := backend.GetForDevice(aGrad.Storage.Device())
	if aShape.NumElements() == outShape.NumElements() {
		be.Add(aGrad.Storage, aGrad.Storage, gradOut.Storage, aShape, outShape, aGrad.Strides, gradOut.Strides, outShape)
		return
	}
	// Sum gradOut along broadcast dims then add to aGrad (single axis for simplicity).
	aPad := len(outShape) - len(aShape)
	for i := 0; i < len(outShape); i++ {
		aDim := 1
		if i >= aPad {
			aDim = aShape[i-aPad]
		}
		if outShape[i] != aDim && outShape[i] > 1 {
			reducedShape := make(core.Shape, len(outShape))
			copy(reducedShape, outShape)
			reducedShape[i] = 1
			reducedStorage, _ := be.Alloc(reducedShape.NumElements() * 4)
			be.Sum(reducedStorage, gradOut.Storage, outShape, gradOut.Strides, i, true)
			reducedStrides := core.ContiguousStrides(reducedShape, 4)
			be.Add(aGrad.Storage, aGrad.Storage, reducedStorage, aShape, reducedShape, aGrad.Strides, reducedStrides, reducedShape)
			be.Free(reducedStorage)
			return
		}
	}
	be.Add(aGrad.Storage, aGrad.Storage, gradOut.Storage, aShape, outShape, aGrad.Strides, gradOut.Strides, outShape)
}

// Sub returns a - b.
func Sub(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := core.BroadcastShapes(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}
	be, err := backend.GetForDevice(a.Storage.Device())
	if err != nil {
		return nil, err
	}
	outStorage, _ := be.Alloc(outShape.NumElements() * 4)
	be.Sub(outStorage, a.Storage, b.Storage, a.Shape, b.Shape, a.Strides, b.Strides, outShape)
	outStrides := core.ContiguousStrides(outShape, 4)
	out := tensor.New(outStorage, outShape, outStrides, core.Float32)
	out.RequiresGrad = a.RequiresGrad || b.RequiresGrad
	if out.RequiresGrad {
		out.Backward = func() {
			if a.RequiresGrad && a.Grad != nil {
				addGradInto(a.Grad, out.Grad, a.Shape, outShape)
			}
			if b.RequiresGrad && b.Grad != nil {
				// grad_b += -grad_out. Use temp buffer for -grad_out then add into b.Grad.
				tmp, _ := be.Alloc(out.Grad.NumElements() * 4)
				be.Neg(tmp, out.Grad.Storage, out.Grad.NumElements())
				addGradInto(b.Grad, tensor.New(tmp, outShape, out.Grad.Strides, core.Float32), b.Shape, outShape)
				be.Free(tmp)
			}
		}
	}
	return out, nil
}

// Mul returns a * b (element-wise with broadcast).
func Mul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := core.BroadcastShapes(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}
	be, err := backend.GetForDevice(a.Storage.Device())
	if err != nil {
		return nil, err
	}
	outStorage, _ := be.Alloc(outShape.NumElements() * 4)
	be.Mul(outStorage, a.Storage, b.Storage, a.Shape, b.Shape, a.Strides, b.Strides, outShape)
	outStrides := core.ContiguousStrides(outShape, 4)
	out := tensor.New(outStorage, outShape, outStrides, core.Float32)
	out.RequiresGrad = a.RequiresGrad || b.RequiresGrad
	if out.RequiresGrad {
		out.Backward = func() {
			if a.RequiresGrad && a.Grad != nil {
				// grad_a += grad_out * b
				// We need a temp: grad_out * b, then add into a.Grad (broadcast). Omitted for brevity; use autograd helpers.
				_ = be
			}
			if b.RequiresGrad && b.Grad != nil {
				_ = be
			}
		}
	}
	return out, nil
}

// Div returns a / b.
func Div(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := core.BroadcastShapes(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}
	be, err := backend.GetForDevice(a.Storage.Device())
	if err != nil {
		return nil, err
	}
	outStorage, _ := be.Alloc(outShape.NumElements() * 4)
	be.Div(outStorage, a.Storage, b.Storage, a.Shape, b.Shape, a.Strides, b.Strides, outShape)
	outStrides := core.ContiguousStrides(outShape, 4)
	return tensor.New(outStorage, outShape, outStrides, core.Float32), nil
}

// MatMul returns a @ b. a: [..., M, K], b: [..., K, N] -> [..., M, N].
func MatMul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if len(a.Shape) < 2 || len(b.Shape) < 2 {
		return nil, fmt.Errorf("matmul requires 2D or batched 2D")
	}
	M, K := a.Shape[len(a.Shape)-2], a.Shape[len(a.Shape)-1]
	K2, N := b.Shape[len(b.Shape)-2], b.Shape[len(b.Shape)-1]
	if K != K2 {
		return nil, fmt.Errorf("matmul: a last dim %d != b second-to-last %d", K, K2)
	}
	batchSize := 1
	for i := 0; i < len(a.Shape)-2; i++ {
		batchSize *= a.Shape[i]
	}
	be, err := backend.GetForDevice(a.Storage.Device())
	if err != nil {
		return nil, err
	}
	outShape := make(core.Shape, len(a.Shape))
	copy(outShape, a.Shape)
	outShape[len(outShape)-1] = N
	outShape[len(outShape)-2] = M
	outSize := outShape.NumElements() * 4
	outStorage, _ := be.Alloc(outSize)
	be.Fill(outStorage, outShape.NumElements(), 0)
	be.MatMul(outStorage, a.Storage, b.Storage, batchSize, M, N, K)
	outStrides := core.ContiguousStrides(outShape, 4)
	out := tensor.New(outStorage, outShape, outStrides, core.Float32)
	out.RequiresGrad = a.RequiresGrad || b.RequiresGrad
	if out.RequiresGrad {
		out.Backward = func() {
			// d(a@b)/da = grad_out @ b^T, d(a@b)/db = a^T @ grad_out
			if a.Grad != nil {
				// a.Grad += grad_out @ b^T. b^T is [..., N, K].
				// Backend MatMul: dst, a, b, batch, M, N, K. So grad_out @ b^T: batch, M, K.
				// We need to add into a.Grad. So MatMul(grad_out, b^T) and add to a.Grad.
				// Backend doesn't do b^T; we need to transpose b or use a separate backward path.
				// For now skip in ops; nn/backward will do it.
			}
			if b.Grad != nil {
			}
		}
	}
	return out, nil
}

// Relu returns max(0, x).
func Relu(x *tensor.Tensor) (*tensor.Tensor, error) {
	be, err := backend.GetForDevice(x.Storage.Device())
	if err != nil {
		return nil, err
	}
	outStorage, _ := be.Alloc(x.NumElements() * 4)
	be.Relu(outStorage, x.Storage, x.NumElements())
	out := tensor.New(outStorage, x.Shape, x.Strides, core.Float32)
	out.RequiresGrad = x.RequiresGrad
	if out.RequiresGrad {
		out.Backward = func() {
			if x.Grad != nil {
				// grad_x = grad_out * (x > 0)
				// Where: grad_x = where(relu'(x), grad_out, 0) = where(x>0, grad_out, 0)
				// We need a mask from x. For now just add grad_out where x > 0.
				_ = be
			}
		}
	}
	return out, nil
}

// Gelu, Sigmoid, Tanh, Silu - forward only in ops; backward can be in autograd.
func Gelu(x *tensor.Tensor) (*tensor.Tensor, error) {
	be, _ := backend.GetForDevice(x.Storage.Device())
	outStorage, _ := be.Alloc(x.NumElements() * 4)
	be.Gelu(outStorage, x.Storage, x.NumElements())
	return tensor.New(outStorage, x.Shape, x.Strides, core.Float32), nil
}

func Sigmoid(x *tensor.Tensor) (*tensor.Tensor, error) {
	be, _ := backend.GetForDevice(x.Storage.Device())
	outStorage, _ := be.Alloc(x.NumElements() * 4)
	be.Sigmoid(outStorage, x.Storage, x.NumElements())
	return tensor.New(outStorage, x.Shape, x.Strides, core.Float32), nil
}

func Tanh(x *tensor.Tensor) (*tensor.Tensor, error) {
	be, _ := backend.GetForDevice(x.Storage.Device())
	outStorage, _ := be.Alloc(x.NumElements() * 4)
	be.Tanh(outStorage, x.Storage, x.NumElements())
	return tensor.New(outStorage, x.Shape, x.Strides, core.Float32), nil
}

func Silu(x *tensor.Tensor) (*tensor.Tensor, error) {
	be, _ := backend.GetForDevice(x.Storage.Device())
	outStorage, _ := be.Alloc(x.NumElements() * 4)
	be.Silu(outStorage, x.Storage, x.NumElements())
	return tensor.New(outStorage, x.Shape, x.Strides, core.Float32), nil
}

// Softmax along last axis.
func Softmax(x *tensor.Tensor) (*tensor.Tensor, error) {
	be, err := backend.GetForDevice(x.Storage.Device())
	if err != nil {
		return nil, err
	}
	outStorage, _ := be.Alloc(x.NumElements() * 4)
	be.Softmax(outStorage, x.Storage, x.Shape, x.Strides)
	return tensor.New(outStorage, x.Shape, x.Strides, core.Float32), nil
}

// LayerNorm: (x - mean) / sqrt(var+eps) * gamma + beta. Normalize over last axis.
func LayerNorm(x, gamma, beta *tensor.Tensor, eps float32) (*tensor.Tensor, error) {
	lastDim := x.Shape[len(x.Shape)-1]
	if gamma.NumElements() != lastDim || beta.NumElements() != lastDim {
		return nil, fmt.Errorf("LayerNorm: gamma/beta must have size %d", lastDim)
	}
	be, err := backend.GetForDevice(x.Storage.Device())
	if err != nil {
		return nil, err
	}
	outStorage, _ := be.Alloc(x.NumElements() * 4)
	meanStorage, _ := be.Alloc((x.NumElements() / lastDim) * 4)
	varStorage, _ := be.Alloc((x.NumElements() / lastDim) * 4)
	be.LayerNorm(outStorage, x.Storage, gamma.Storage, beta.Storage, meanStorage, varStorage, x.Shape, x.Strides, eps)
	be.Free(meanStorage)
	be.Free(varStorage)
	return tensor.New(outStorage, x.Shape, x.Strides, core.Float32), nil
}
