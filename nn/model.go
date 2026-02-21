package nn

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// LLM is Embedding -> TransformerBlock × N -> LayerNorm -> Linear (output head).
type LLM struct {
	Embedding   *Embedding
	Blocks      []*TransformerBlock
	FinalNorm   *LayerNorm
	OutputHead  *Linear
	VocabSize   int
	EmbedDim    int
	NumHeads    int
	HeadDim     int
	NumLayers   int
	MaxSeqLen   int
}

// NewLLM builds an LLM from pre-built components.
func NewLLM(emb *Embedding, blocks []*TransformerBlock, finalNorm *LayerNorm, outHead *Linear, vocabSize, embedDim, numLayers, maxSeqLen int) *LLM {
	numHeads := 0
	headDim := 0
	if len(blocks) > 0 && blocks[0].Attn != nil {
		numHeads = blocks[0].Attn.NumHeads
		headDim = blocks[0].Attn.HeadDim
	}
	return &LLM{
		Embedding:  emb,
		Blocks:    blocks,
		FinalNorm: finalNorm,
		OutputHead: outHead,
		VocabSize:  vocabSize,
		EmbedDim:   embedDim,
		NumHeads:   numHeads,
		HeadDim:    headDim,
		NumLayers:  len(blocks),
		MaxSeqLen:  maxSeqLen,
	}
}

// Forward runs the full model. indices [batch, seq] int64 -> logits [batch, seq, vocabSize].
func (m *LLM) Forward(indices *tensor.Tensor) (*tensor.Tensor, error) {
	if indices.DType != core.Int64 {
		return nil, fmt.Errorf("indices must be int64")
	}
	if len(indices.Shape) != 2 {
		return nil, fmt.Errorf("indices must be [batch, seq]")
	}
	batch, seq := indices.Shape[0], indices.Shape[1]
	x, err := m.Embedding.Forward(indices)
	if err != nil {
		return nil, err
	}
	// x [batch, seq, embedDim]
	for _, blk := range m.Blocks {
		x, err = blk.Forward(x)
		if err != nil {
			return nil, err
		}
	}
	normOut, err := m.FinalNorm.Forward(x)
	if err != nil {
		return nil, err
	}
	// normOut [batch, seq, embedDim] -> OutputHead -> [batch, seq, vocabSize]
	logits, err := m.OutputHead.Forward(normOut)
	if err != nil {
		return nil, err
	}
	_ = batch
	_ = seq
	return logits, nil
}

// InitSmall allocates and initializes a small LLM for testing.
func InitSmall(vocabSize, embedDim, numHeads, numLayers, maxSeqLen int) (*LLM, error) {
	be, err := backend.GetForDevice(backend.CPU0)
	if err != nil {
		return nil, err
	}
	headDim := embedDim / numHeads
	if headDim*numHeads != embedDim {
		return nil, fmt.Errorf("embedDim must be divisible by numHeads")
	}
	// Embedding table [vocabSize, embedDim]
	embTable, _ := be.Alloc(vocabSize * embedDim * 4)
	be.Fill(embTable, vocabSize*embedDim, 0.01)
	embShape := core.Shape{vocabSize, embedDim}
	embStrides := core.ContiguousStrides(embShape, 4)
	embTensor := tensor.New(embTable, embShape, embStrides, core.Float32)
	emb, _ := NewEmbedding(embTensor)
	// Blocks
	blocks := make([]*TransformerBlock, numLayers)
	hidden := embedDim * 4 // FFN hidden
	for L := 0; L < numLayers; L++ {
		// Q,K,V,Out projections: in=embedDim, out=embedDim
		allocLin := func(in, out int) (*Linear, error) {
			w, _ := be.Alloc(out * in * 4)
			be.Fill(w, out*in, 0.02)
			b, _ := be.Alloc(out * 4)
			be.Fill(b, out, 0)
			wT := tensor.New(w, core.Shape{out, in}, core.ContiguousStrides(core.Shape{out, in}, 4), core.Float32)
			bT := tensor.New(b, core.Shape{out}, core.ContiguousStrides(core.Shape{out}, 4), core.Float32)
			return NewLinear(in, out, wT, bT)
		}
		q, _ := allocLin(embedDim, embedDim)
		k, _ := allocLin(embedDim, embedDim)
		v, _ := allocLin(embedDim, embedDim)
		outProj, _ := allocLin(embedDim, embedDim)
		attn := NewAttention(numHeads, headDim, q, k, v, outProj, 10000)
		// FFN: embedDim -> hidden -> embedDim
		w1, _ := be.Alloc(hidden * embedDim * 4)
		be.Fill(w1, hidden*embedDim, 0.02)
		b1, _ := be.Alloc(hidden * 4)
		be.Fill(b1, hidden, 0)
		w2, _ := be.Alloc(embedDim * hidden * 4)
		be.Fill(w2, embedDim*hidden, 0.02)
		b2, _ := be.Alloc(embedDim * 4)
		be.Fill(b2, embedDim, 0)
		lin1, _ := NewLinear(embedDim, hidden, tensor.New(w1, core.Shape{hidden, embedDim}, core.ContiguousStrides(core.Shape{hidden, embedDim}, 4), core.Float32), tensor.New(b1, core.Shape{hidden}, core.ContiguousStrides(core.Shape{hidden}, 4), core.Float32))
		lin2, _ := NewLinear(hidden, embedDim, tensor.New(w2, core.Shape{embedDim, hidden}, core.ContiguousStrides(core.Shape{embedDim, hidden}, 4), core.Float32), tensor.New(b2, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32))
		ffn := &FeedForward{Linear1: lin1, Linear2: lin2}
		gamma, _ := be.Alloc(embedDim * 4)
		be.Fill(gamma, embedDim, 1)
		beta, _ := be.Alloc(embedDim * 4)
		be.Fill(beta, embedDim, 0)
		gT := tensor.New(gamma, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32)
		bT := tensor.New(beta, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32)
		n1 := NewLayerNorm(gT, bT, 1e-5)
		gamma2, _ := be.Alloc(embedDim * 4)
		be.Fill(gamma2, embedDim, 1)
		beta2, _ := be.Alloc(embedDim * 4)
		be.Fill(beta2, embedDim, 0)
		n2 := NewLayerNorm(tensor.New(gamma2, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32), tensor.New(beta2, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32), 1e-5)
		blocks[L] = &TransformerBlock{Attn: attn, FFN: ffn, Norm1: n1, Norm2: n2}
	}
	// Final norm
	gammaF, _ := be.Alloc(embedDim * 4)
	be.Fill(gammaF, embedDim, 1)
	betaF, _ := be.Alloc(embedDim * 4)
	be.Fill(betaF, embedDim, 0)
	finalNorm := NewLayerNorm(
		tensor.New(gammaF, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32),
		tensor.New(betaF, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32),
		1e-5)
	// Output head: embedDim -> vocabSize
	wOut, _ := be.Alloc(vocabSize * embedDim * 4)
	be.Fill(wOut, vocabSize*embedDim, 0.02)
	bOut, _ := be.Alloc(vocabSize * 4)
	be.Fill(bOut, vocabSize, 0)
	outHead, _ := NewLinear(embedDim, vocabSize, tensor.New(wOut, core.Shape{vocabSize, embedDim}, core.ContiguousStrides(core.Shape{vocabSize, embedDim}, 4), core.Float32), tensor.New(bOut, core.Shape{vocabSize}, core.ContiguousStrides(core.Shape{vocabSize}, 4), core.Float32))
	return NewLLM(emb, blocks, finalNorm, outHead, vocabSize, embedDim, numLayers, maxSeqLen), nil
}
