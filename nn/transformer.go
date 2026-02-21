package nn

import (
	"github.com/djeday123/goml/ops"
	"github.com/djeday123/goml/tensor"
)

// TransformerBlock is Attention + residual + LayerNorm + FeedForward + residual + LayerNorm.
type TransformerBlock struct {
	Attn       *Attention
	FFN        *FeedForward
	Norm1      *LayerNorm
	Norm2      *LayerNorm
}

// Forward runs the block. x [batch, seq, dim].
func (b *TransformerBlock) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	attnOut, err := b.Attn.Forward(x)
	if err != nil {
		return nil, err
	}
	// residual + norm
	res1, err := ops.Add(x, attnOut)
	if err != nil {
		return nil, err
	}
	norm1Out, err := b.Norm1.Forward(res1)
	if err != nil {
		return nil, err
	}
	ffnOut, err := b.FFN.Forward(norm1Out)
	if err != nil {
		return nil, err
	}
	res2, err := ops.Add(norm1Out, ffnOut)
	if err != nil {
		return nil, err
	}
	return b.Norm2.Forward(res2)
}
