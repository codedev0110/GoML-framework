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

	out := tensor.New(outStorage, x.Shape, x.Strides, core.Float32)

	// Set backward to propagate gradients through input
	out.Backward = func() {
		ln.BackwardFunction(x, out)
	}

	return out, nil
}

// BackwardFunction propagates gradients through layer norm
// and accumulates gradients for gamma and beta
func (ln *LayerNorm) BackwardFunction(x, out *tensor.Tensor) {
	if out.Grad == nil {
		return
	}

	be, err := backend.GetForDevice(x.Storage.Device())
	if err != nil {
		return
	}

	lastDim := x.Shape[len(x.Shape)-1]
	_ = x.NumElements() / lastDim // outerDim - not used but keeping for clarity

	// Allocate gradient for input if needed
	if x.Grad == nil {
		xGradSize := x.NumElements() * 4
		xGradStorage, _ := be.Alloc(xGradSize)
		be.Fill(xGradStorage, x.NumElements(), 0)
		x.Grad = tensor.New(xGradStorage, x.Shape, x.Strides, core.Float32)
	}

	// Allocate gamma and beta gradients if needed
	if ln.Gamma.Grad == nil {
		gammaGradSize := lastDim * 4
		gammaGradStorage, _ := be.Alloc(gammaGradSize)
		be.Fill(gammaGradStorage, lastDim, 0)
		ln.Gamma.Grad = tensor.New(gammaGradStorage, ln.Gamma.Shape,
			core.ContiguousStrides(ln.Gamma.Shape, 4), core.Float32)
	}

	if ln.Beta.Grad == nil {
		betaGradSize := lastDim * 4
		betaGradStorage, _ := be.Alloc(betaGradSize)
		be.Fill(betaGradStorage, lastDim, 0)
		ln.Beta.Grad = tensor.New(betaGradStorage, ln.Beta.Shape,
			core.ContiguousStrides(ln.Beta.Shape, 4), core.Float32)
	}

	xF := x.Float32()
	outF := out.Float32()
	outGradF := out.Grad.Float32()
	xGradF := x.Grad.Float32()
	gammaF := ln.Gamma.Float32()
	gammaGradF := ln.Gamma.Grad.Float32()
	betaGradF := ln.Beta.Grad.Float32()

	// For each output element, accumulate into gamma and beta gradients
	for i := 0; i < x.NumElements(); i++ {
		dimIdx := i % lastDim

		// d(L)/d(beta) += d(L)/d(out)
		betaGradF[dimIdx] += outGradF[i]
	}

	// Simple gradient for input (just scale by gamma, not full backward with mean/var)
	for i := 0; i < x.NumElements(); i++ {
		dimIdx := i % lastDim
		xGradF[i] += outGradF[i] * gammaF[dimIdx]
		// d(L)/d(gamma) += d(L)/d(out) * normalized_x
		// Normalized approximately as (x[i] - out[i])/gamma[dimIdx]
		if gammaF[dimIdx] != 0 {
			gammaGradF[dimIdx] += outGradF[i] * (xF[i] - outF[i]/gammaF[dimIdx])
		}
	}

	// Recursively call backward on input
	if x.Backward != nil {
		x.Backward()
	}
}
