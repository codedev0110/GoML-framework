package nn

import (
	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// LayerNorm normalizes over the last dimension: (x - mean) / sqrt(var + eps) * gamma + beta.
type LayerNorm struct {
	Gamma *tensor.Tensor // same size as last dim
	Beta  *tensor.Tensor
	Eps   float32
}

// NewLayerNorm creates a LayerNorm with learnable gamma and beta.
func NewLayerNorm(gamma, beta *tensor.Tensor, eps float32) *LayerNorm {
	if eps == 0 {
		eps = 1e-5
	}
	return &LayerNorm{Gamma: gamma, Beta: beta, Eps: eps}
}

// Forward applies layer normalization.
func (ln *LayerNorm) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	be, err := backend.GetForDevice(x.Storage.Device())
	if err != nil {
		return nil, err
	}
	lastDim := x.Shape[len(x.Shape)-1]
	outStorage, _ := be.Alloc(x.NumElements() * 4)
	meanStorage, _ := be.Alloc((x.NumElements() / lastDim) * 4)
	varStorage, _ := be.Alloc((x.NumElements() / lastDim) * 4)
	be.LayerNorm(outStorage, x.Storage, ln.Gamma.Storage, ln.Beta.Storage, meanStorage, varStorage, x.Shape, x.Strides, ln.Eps)
	be.Free(meanStorage)
	be.Free(varStorage)
	return tensor.New(outStorage, x.Shape, x.Strides, core.Float32), nil
}
