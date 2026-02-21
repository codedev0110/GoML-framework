package nn

import (
	"fmt"
	"unsafe"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// Attention is multi-head self-attention with RoPE and causal mask.
type Attention struct {
	NumHeads int
	HeadDim  int
	QProj    *Linear
	KProj    *Linear
	VProj    *Linear
	OutProj  *Linear
	RoPEBase float64
}

// NewAttention creates multi-head attention. inDim = numHeads * headDim.
func NewAttention(numHeads, headDim int, q, k, v, out *Linear, ropeBase float64) *Attention {
	if ropeBase == 0 {
		ropeBase = 10000
	}
	return &Attention{
		NumHeads: numHeads,
		HeadDim:  headDim,
		QProj:    q,
		KProj:    k,
		VProj:    v,
		OutProj:  out,
		RoPEBase: ropeBase,
	}
}

// Forward runs attention. x [batch, seq, inDim]. Returns [batch, seq, inDim].
func (a *Attention) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	batch := x.Shape[0]
	seq := x.Shape[1]
	inDim := a.NumHeads * a.HeadDim
	if len(x.Shape) < 3 || x.Shape[2] != inDim {
		return nil, fmt.Errorf("attention: x last dim must be %d", inDim)
	}
	be, err := backend.GetForDevice(x.Storage.Device())
	if err != nil {
		return nil, err
	}
	q, err := a.QProj.Forward(x)
	if err != nil {
		return nil, err
	}
	k, err := a.KProj.Forward(x)
	if err != nil {
		return nil, err
	}
	v, err := a.VProj.Forward(x)
	if err != nil {
		return nil, err
	}
	// q,k,v are [batch*seq, inDim]. Permute to [batch, numHeads, seq, headDim] for RoPE and SDPA.
	qShape := core.Shape{batch, a.NumHeads, seq, a.HeadDim}
	qStrides := core.ContiguousStrides(qShape, 4)
	ropeQ, _ := be.Alloc(batch * a.NumHeads * seq * a.HeadDim * 4)
	ropeK, _ := be.Alloc(batch * a.NumHeads * seq * a.HeadDim * 4)
	vHSD, _ := be.Alloc(batch * a.NumHeads * seq * a.HeadDim * 4)
	permuteBSEQToBHSD(ropeQ, q.Storage, batch, seq, a.NumHeads, a.HeadDim)
	permuteBSEQToBHSD(ropeK, k.Storage, batch, seq, a.NumHeads, a.HeadDim)
	permuteBSEQToBHSD(vHSD, v.Storage, batch, seq, a.NumHeads, a.HeadDim)
	be.RoPE(ropeQ, ropeQ, qShape, qStrides, a.RoPEBase, 0, seq)
	be.RoPE(ropeK, ropeK, qShape, qStrides, a.RoPEBase, 0, seq)
	attnOutSize := batch * a.NumHeads * seq * a.HeadDim
	attnOut, _ := be.Alloc(attnOutSize * 4)
	be.ScaledDotProductAttention(attnOut, ropeQ, ropeK, vHSD, batch, a.NumHeads, seq, a.HeadDim, true)
	be.Free(vHSD)
	be.Free(ropeQ)
	be.Free(ropeK)
	// Permute [batch, heads, seq, headDim] -> [batch, seq, inDim] for out projection
	outStorage, _ := be.Alloc(batch * seq * inDim * 4)
	permuteBHSDToBSHD(outStorage, attnOut, batch, a.NumHeads, seq, a.HeadDim)
	be.Free(attnOut)
	outShape := core.Shape{batch, seq, inDim}
	outStrides := core.ContiguousStrides(outShape, 4)
	outTensor := tensor.New(outStorage, outShape, outStrides, core.Float32)
	return a.OutProj.Forward(outTensor)
}

// permuteBSEQToBHSD copies [batch, seq, embedDim] to [batch, heads, seq, headDim] (embedDim = heads*headDim).
func permuteBSEQToBHSD(dst, src backend.Storage, batch, seq, heads, headDim int) {
	dstB := dst.Bytes()
	srcB := src.Bytes()
	if dstB == nil || srcB == nil {
		return
	}
	dstF := unsafe.Slice((*float32)(unsafe.Pointer(&dstB[0])), len(dstB)/4)
	srcF := unsafe.Slice((*float32)(unsafe.Pointer(&srcB[0])), len(srcB)/4)
	embedDim := heads * headDim
	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			for h := 0; h < heads; h++ {
				for d := 0; d < headDim; d++ {
					srcIdx := (b*seq+s)*embedDim + h*headDim + d
					dstIdx := ((b*heads+h)*seq+s)*headDim + d
					dstF[dstIdx] = srcF[srcIdx]
				}
			}
		}
	}
}

// permuteBHSDToBSHD copies [batch, heads, seq, headDim] to [batch, seq, heads*headDim] (row-major).
func permuteBHSDToBSHD(dst, src backend.Storage, batch, heads, seq, headDim int) {
	dstB := dst.Bytes()
	srcB := src.Bytes()
	if dstB == nil || srcB == nil {
		return
	}
	dstF := unsafe.Slice((*float32)(unsafe.Pointer(&dstB[0])), len(dstB)/4)
	srcF := unsafe.Slice((*float32)(unsafe.Pointer(&srcB[0])), len(srcB)/4)
	inDim := heads * headDim
	for b := 0; b < batch; b++ {
		for s := 0; s < seq; s++ {
			for h := 0; h < heads; h++ {
				for d := 0; d < headDim; d++ {
					srcIdx := ((b*heads+h)*seq+s)*headDim + d
					dstIdx := (b*seq+s)*inDim + h*headDim + d
					dstF[dstIdx] = srcF[srcIdx]
				}
			}
		}
	}
}
