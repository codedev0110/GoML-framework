package ops

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// Add returns a + b with broadcasting.  If either operand requires
// gradients the returned tensor will carry a grad function that accumulates
// into the inputs.
func Add(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := core.BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	be, err := backend.GetForDevice(a.Storage().Device())
	if err != nil {
		return nil, err
	}
	byteLen := outShape.NumElements() * int(core.Float32.Size())
	outStorage, err := be.Alloc(byteLen)
	if err != nil {
		return nil, err
	}
	if err := be.Add(outStorage, a.Storage(), b.Storage(), a.Shape(), b.Shape(), a.Strides(), b.Strides(), outShape); err != nil {
		outStorage.Free()
		return nil, err
	}
	out := tensor.NewTensor(outStorage, outShape, core.Float32)
	out.SetRequiresGrad(a.RequiresGrad() || b.RequiresGrad())
	if out.RequiresGrad() {
		out.SetGradFn(func() {
			if a.RequiresGrad() && a.Grad() != nil {
				addGradInto(a.Grad(), out.Grad(), a.Shape(), outShape)
			}
			if b.RequiresGrad() && b.Grad() != nil {
				addGradInto(b.Grad(), out.Grad(), b.Shape(), outShape)
			}
		})
	}
	return out, nil
}

// addGradInto adds gradOut (broadcast/sum to match aShape) into aGrad.
func addGradInto(aGrad, gradOut *tensor.Tensor, aShape, outShape core.Shape) {
	be, _ := backend.GetForDevice(aGrad.Storage().Device())
	if aShape.NumElements() == outShape.NumElements() {
		be.Add(aGrad.Storage(), aGrad.Storage(), gradOut.Storage(), aShape, outShape, aGrad.Strides(), gradOut.Strides(), outShape)
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
			reducedStorage, _ := be.Alloc(reducedShape.NumElements() * int(core.Float32.Size()))
			be.Sum(reducedStorage, gradOut.Storage(), outShape, gradOut.Strides(), i, true)
			reducedStrides := core.ContiguousStrides(reducedShape, uintptr(core.Float32.Size()))
			be.Add(aGrad.Storage(), aGrad.Storage(), reducedStorage, aShape, reducedShape, aGrad.Strides(), reducedStrides, reducedShape)
			be.Free(reducedStorage)
			return
		}
	}
	be.Add(aGrad.Storage(), aGrad.Storage(), gradOut.Storage(), aShape, outShape, aGrad.Strides(), gradOut.Strides(), outShape)
}

// Sub returns a - b.
func Sub(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := core.BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	be, err := backend.GetForDevice(a.Storage().Device())
	if err != nil {
		return nil, err
	}
	outStorage, _ := be.Alloc(outShape.NumElements() * int(core.Float32.Size()))
	be.Sub(outStorage, a.Storage(), b.Storage(), a.Shape(), b.Shape(), a.Strides(), b.Strides(), outShape)
	out := tensor.NewTensor(outStorage, outShape, core.Float32)
	out.SetRequiresGrad(a.RequiresGrad() || b.RequiresGrad())
	if out.RequiresGrad() {
		out.SetGradFn(func() {
			if a.RequiresGrad() && a.Grad() != nil {
				addGradInto(a.Grad(), out.Grad(), a.Shape(), outShape)
			}
			if b.RequiresGrad() && b.Grad() != nil {
				// grad_b += -grad_out. Use temp buffer for -grad_out then add into b.Grad.
				tmp, _ := be.Alloc(out.Grad().NumElements() * int(core.Float32.Size()))
				be.Neg(tmp, out.Grad().Storage(), out.Grad().NumElements())
				addGradInto(b.Grad(), tensor.NewTensor(tmp, outShape, core.Float32), b.Shape(), outShape)
				be.Free(tmp)
			}
		})
	}
	return out, nil
}

// Mul returns a * b (element-wise with broadcast).
func Mul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := core.BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	be, err := backend.GetForDevice(a.Storage().Device())
	if err != nil {
		return nil, err
	}
	outStorage, _ := be.Alloc(outShape.NumElements() * int(core.Float32.Size()))
	be.Mul(outStorage, a.Storage(), b.Storage(), a.Shape(), b.Shape(), a.Strides(), b.Strides(), outShape)
	out := tensor.NewTensor(outStorage, outShape, core.Float32)
	out.SetRequiresGrad(a.RequiresGrad() || b.RequiresGrad())
	if out.RequiresGrad() {
		out.SetGradFn(func() {
			if a.RequiresGrad() && a.Grad() != nil {
				// grad_a += grad_out * b
				_ = be
			}
			if b.RequiresGrad() && b.Grad() != nil {
				_ = be
			}
		})
	}
	return out, nil
}

// Div returns a / b.
func Div(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	outShape, err := core.BroadcastShapes(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	be, err := backend.GetForDevice(a.Storage().Device())
	if err != nil {
		return nil, err
	}
	outStorage, _ := be.Alloc(outShape.NumElements() * int(core.Float32.Size()))
	be.Div(outStorage, a.Storage(), b.Storage(), a.Shape(), b.Shape(), a.Strides(), b.Strides(), outShape)
	return tensor.NewTensor(outStorage, outShape, core.Float32), nil
}

// MatMul returns a @ b. a: [..., M, K], b: [..., K, N] -> [..., M, N].
func MatMul(a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if a.NDim() < 2 || b.NDim() < 2 {
		return nil, fmt.Errorf("matmul requires 2D or batched 2D")
	}
	M, K := a.Shape()[a.NDim()-2], a.Shape()[a.NDim()-1]
	K2, N := b.Shape()[b.NDim()-2], b.Shape()[b.NDim()-1]
	if K != K2 {
		return nil, fmt.Errorf("matmul: a last dim %d != b second-to-last %d", K, K2)
	}
	batchSize := 1
	for i := 0; i < a.NDim()-2; i++ {
		batchSize *= a.Shape()[i]
	}
	be, err := backend.GetForDevice(a.Storage().Device())
	if err != nil {
		return nil, err
	}
	outShape := make(core.Shape, a.NDim())
	copy(outShape, a.Shape())
	outShape[a.NDim()-1] = N
	outShape[a.NDim()-2] = M
	outSize := outShape.NumElements() * int(core.Float32.Size())
	outStorage, _ := be.Alloc(outSize)
	be.Fill(outStorage, outShape.NumElements(), 0)
	be.MatMul(outStorage, a.Storage(), b.Storage(), batchSize, M, N, K)
	out := tensor.NewTensor(outStorage, outShape, core.Float32)
	out.SetRequiresGrad(a.RequiresGrad() || b.RequiresGrad())
	if out.RequiresGrad() {
		out.SetGradFn(func() {
			// d(a@b)/da = grad_out @ b^T, d(a@b)/db = a^T @ grad_out
			if a.Grad() != nil {
				// a.Grad += grad_out @ b^T. b^T is [..., N, K].
				// Backend MatMul: dst, a, b, batch, M, N, K. So grad_out @ b^T: batch, M, K.
				// We need to add into a.Grad. So MatMul(grad_out, b^T) and add to a.Grad.
				// Backend doesn't do b^T; we need to transpose b or use a separate backward path.
				// For now skip in ops; nn/backward will do it.
			}
			if b.Grad() != nil {
			}
		})
	}
	return out, nil
}

// Relu returns max(0, x).
func Relu(x *tensor.Tensor) (*tensor.Tensor, error) {
	be, err := backend.GetForDevice(x.Storage().Device())
	if err != nil {
		return nil, err
	}
	outStorage, _ := be.Alloc(x.NumElements() * int(core.Float32.Size()))
	be.Relu(outStorage, x.Storage(), x.NumElements())
	out := tensor.NewTensor(outStorage, x.Shape(), core.Float32)
	out.SetRequiresGrad(x.RequiresGrad())
	if out.RequiresGrad() {
		out.SetGradFn(func() {
			if x.Grad() != nil {
				// grad_x = grad_out * (x > 0)
				_ = be
			}
		})
	}
	return out, nil
}

// Gelu, Sigmoid, Tanh, Silu - forward only in ops; backward can be in autograd.
func Gelu(x *tensor.Tensor) (*tensor.Tensor, error) {
	be, _ := backend.GetForDevice(x.Storage().Device())
	outStorage, _ := be.Alloc(x.NumElements() * int(core.Float32.Size()))
	be.Gelu(outStorage, x.Storage(), x.NumElements())
	return tensor.NewTensor(outStorage, x.Shape(), core.Float32), nil
}

func Sigmoid(x *tensor.Tensor) (*tensor.Tensor, error) {
	be, _ := backend.GetForDevice(x.Storage().Device())
	outStorage, _ := be.Alloc(x.NumElements() * int(core.Float32.Size()))
	be.Sigmoid(outStorage, x.Storage(), x.NumElements())
	return tensor.NewTensor(outStorage, x.Shape(), core.Float32), nil
}

func Tanh(x *tensor.Tensor) (*tensor.Tensor, error) {
	be, _ := backend.GetForDevice(x.Storage().Device())
	outStorage, _ := be.Alloc(x.NumElements() * int(core.Float32.Size()))
	be.Tanh(outStorage, x.Storage(), x.NumElements())
	return tensor.NewTensor(outStorage, x.Shape(), core.Float32), nil
}

func Silu(x *tensor.Tensor) (*tensor.Tensor, error) {
	be, _ := backend.GetForDevice(x.Storage().Device())
	outStorage, _ := be.Alloc(x.NumElements() * int(core.Float32.Size()))
	be.Silu(outStorage, x.Storage(), x.NumElements())
	return tensor.NewTensor(outStorage, x.Shape(), core.Float32), nil
}

// Softmax along last axis.
func Softmax(x *tensor.Tensor) (*tensor.Tensor, error) {
	be, err := backend.GetForDevice(x.Storage().Device())
	if err != nil {
		return nil, err
	}
	outStorage, _ := be.Alloc(x.NumElements() * int(core.Float32.Size()))
	be.Softmax(outStorage, x.Storage(), x.Shape(), x.Strides())
	return tensor.NewTensor(outStorage, x.Shape(), core.Float32), nil
}

// LayerNorm: (x - mean) / sqrt(var+eps) * gamma + beta. Normalize over last axis.
func LayerNorm(x, gamma, beta *tensor.Tensor, eps float32) (*tensor.Tensor, error) {
	lastDim := x.Shape()[len(x.Shape())-1]
	if gamma.NumElements() != lastDim || beta.NumElements() != lastDim {
		return nil, fmt.Errorf("LayerNorm: gamma/beta must have size %d", lastDim)
	}
	be, err := backend.GetForDevice(x.Storage().Device())
	if err != nil {
		return nil, err
	}
	outStorage, _ := be.Alloc(x.NumElements() * int(core.Float32.Size()))
	meanStorage, _ := be.Alloc((x.NumElements() / lastDim) * int(core.Float32.Size()))
	varStorage, _ := be.Alloc((x.NumElements() / lastDim) * int(core.Float32.Size()))
	be.LayerNorm(outStorage, x.Storage(), gamma.Storage(), beta.Storage(), meanStorage, varStorage, x.Shape(), x.Strides(), eps)
	be.Free(meanStorage)
	be.Free(varStorage)
	return tensor.NewTensor(outStorage, x.Shape(), core.Float32), nil
}
