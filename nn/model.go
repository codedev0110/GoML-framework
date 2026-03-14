package nn

import (
	"fmt"

	"github.com/djeday123/goml/backend"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/tensor"
)

// LLM is Embedding -> TransformerBlock × N -> LayerNorm -> Linear (output head).
type LLM struct {
	Embedding  *Embedding
	Blocks     []*TransformerBlock
	FinalNorm  *LayerNorm
	OutputHead *Linear
	VocabSize  int
	EmbedDim   int
	NumHeads   int
	HeadDim    int
	NumLayers  int
	MaxSeqLen  int
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
		Blocks:     blocks,
		FinalNorm:  finalNorm,
		OutputHead: outHead,
		VocabSize:  vocabSize,
		EmbedDim:   embedDim,
		NumHeads:   numHeads,
		HeadDim:    headDim,
		NumLayers:  len(blocks),
		MaxSeqLen:  maxSeqLen,
	}
}

// Parameters returns all trainable parameter tensors in the model.
func (m *LLM) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	// Embedding table
	params = append(params, m.Embedding.Table)
	// Transformer blocks
	for _, blk := range m.Blocks {
		// Attention projections
		params = append(params, blk.Attn.QProj.W, blk.Attn.QProj.Bias)
		params = append(params, blk.Attn.KProj.W, blk.Attn.KProj.Bias)
		params = append(params, blk.Attn.VProj.W, blk.Attn.VProj.Bias)
		params = append(params, blk.Attn.OutProj.W, blk.Attn.OutProj.Bias)
		// FFN
		params = append(params, blk.FFN.Linear1.W, blk.FFN.Linear1.Bias)
		params = append(params, blk.FFN.Linear2.W, blk.FFN.Linear2.Bias)
		// LayerNorms
		params = append(params, blk.Norm1.Gamma, blk.Norm1.Beta)
		params = append(params, blk.Norm2.Gamma, blk.Norm2.Beta)
	}
	// Final norm
	params = append(params, m.FinalNorm.Gamma, m.FinalNorm.Beta)
	// Output head
	params = append(params, m.OutputHead.W, m.OutputHead.Bias)
	return params
}

// ZeroGrad sets .Grad = nil on all parameters.
func (m *LLM) ZeroGrad() {
	for _, p := range m.Parameters() {
		p.SetGrad(nil)
	}
}

// Forward runs the full model. indices [batch, seq] int64 -> logits [batch, seq, vocabSize].
func (m *LLM) Forward(indices *tensor.Tensor) (*tensor.Tensor, error) {
	if indices.DType() != core.Int64 {
		return nil, fmt.Errorf("indices must be int64")
	}
	if indices.NDim() != 2 {
		return nil, fmt.Errorf("indices must be [batch, seq]")
	}
	batch, seq := indices.Shape()[0], indices.Shape()[1]
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
	embTensor := tensor.NewTensor(embTable, embShape, core.Float32)
	embTensor.SetRequiresGrad(true)
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
			wT := tensor.NewTensor(w, core.Shape{out, in}, core.Float32)
			wT.SetRequiresGrad(true)
			bT := tensor.NewTensor(b, core.Shape{out}, core.Float32)
			bT.SetRequiresGrad(true)
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
		w1T := tensor.NewTensor(w1, core.Shape{hidden, embedDim}, core.Float32)
		w1T.SetRequiresGrad(true)
		b1T := tensor.NewTensor(b1, core.Shape{hidden}, core.Float32)
		b1T.SetRequiresGrad(true)
		lin1, _ := NewLinear(embedDim, hidden, w1T, b1T)
		w2T := tensor.NewTensor(w2, core.Shape{embedDim, hidden}, core.Float32)
		w2T.SetRequiresGrad(true)
		b2T := tensor.NewTensor(b2, core.Shape{embedDim}, core.Float32)
		b2T.SetRequiresGrad(true)
		lin2, _ := NewLinear(hidden, embedDim, w2T, b2T)
		ffn := &FeedForward{Linear1: lin1, Linear2: lin2}
		gamma, _ := be.Alloc(embedDim * 4)
		be.Fill(gamma, embedDim, 1)
		beta, _ := be.Alloc(embedDim * 4)
		be.Fill(beta, embedDim, 0)
		gT := tensor.NewTensor(gamma, core.Shape{embedDim}, core.Float32)
		gT.SetRequiresGrad(true)
		bT := tensor.NewTensor(beta, core.Shape{embedDim}, core.Float32)
		bT.SetRequiresGrad(true)
		n1 := NewLayerNorm(gT, bT, 1e-5)
		gamma2, _ := be.Alloc(embedDim * 4)
		be.Fill(gamma2, embedDim, 1)
		beta2, _ := be.Alloc(embedDim * 4)
		be.Fill(beta2, embedDim, 0)
		g2T := tensor.NewTensor(gamma2, core.Shape{embedDim}, core.Float32)
		g2T.SetRequiresGrad(true)
		b2TN := tensor.NewTensor(beta2, core.Shape{embedDim}, core.Float32)
		b2TN.SetRequiresGrad(true)
		n2 := NewLayerNorm(g2T, b2TN, 1e-5)
		blocks[L] = &TransformerBlock{Attn: attn, FFN: ffn, Norm1: n1, Norm2: n2}
	}
	// Final norm
	gammaF, _ := be.Alloc(embedDim * 4)
	be.Fill(gammaF, embedDim, 1)
	betaF, _ := be.Alloc(embedDim * 4)
	be.Fill(betaF, embedDim, 0)
	gFT := tensor.NewTensor(gammaF, core.Shape{embedDim}, core.Float32)
	gFT.SetRequiresGrad(true)
	bFT := tensor.NewTensor(betaF, core.Shape{embedDim}, core.Float32)
	bFT.SetRequiresGrad(true)
	finalNorm := NewLayerNorm(gFT, bFT, 1e-5)
	// Output head: embedDim -> vocabSize
	wOut, _ := be.Alloc(vocabSize * embedDim * 4)
	be.Fill(wOut, vocabSize*embedDim, 0.02)
	bOut, _ := be.Alloc(vocabSize * 4)
	be.Fill(bOut, vocabSize, 0)
	wOutT := tensor.NewTensor(wOut, core.Shape{vocabSize, embedDim}, core.Float32)
	wOutT.SetRequiresGrad(true)
	bOutT := tensor.NewTensor(bOut, core.Shape{vocabSize}, core.Float32)
	bOutT.SetRequiresGrad(true)
	outHead, _ := NewLinear(embedDim, vocabSize, wOutT, bOutT)
	return NewLLM(emb, blocks, finalNorm, outHead, vocabSize, embedDim, numLayers, maxSeqLen), nil
}
