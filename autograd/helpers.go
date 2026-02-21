package autograd

import (
	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// ZeroGrad allocates gradient storage for t and fills with zero, if RequiresGrad.
func ZeroGrad(t *tensor.Tensor) {
	if !t.RequiresGrad || t.Grad != nil {
		return
	}
	be, err := backend.GetForDevice(t.Storage.Device())
	if err != nil {
		return
	}
	byteLen := t.NumElements() * int(t.DType.Size())
	if t.DType != core.Float32 {
		return
	}
	storage, err := be.Alloc(byteLen)
	if err != nil {
		return
	}
	be.Fill(storage, t.NumElements(), 0)
	t.Grad = tensor.New(storage, t.Shape, t.Strides, t.DType)
}

// AccumulateGrad adds grad into t.Grad (creating t.Grad if nil).
func AccumulateGrad(t *tensor.Tensor, grad *tensor.Tensor) {
	if t.Grad == nil {
		ZeroGrad(t)
	}
	if t.Grad == nil || grad == nil {
		return
	}
	be, _ := backend.GetForDevice(t.Storage.Device())
	be.Add(t.Grad.Storage, t.Grad.Storage, grad.Storage, t.Shape, grad.Shape, t.Grad.Strides, grad.Strides, grad.Shape)
}
