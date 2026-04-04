package nn

import (
	"fmt"
	"math"
	"math/rand"
	"unsafe"

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

// GetParameters collects all trainable parameters (weights and biases).
func (m *LLM) GetParameters() []*tensor.Tensor {
	var params []*tensor.Tensor

	// Embedding parameters
	if m.Embedding != nil && m.Embedding.Table != nil {
		params = append(params, m.Embedding.Table)
	}

	// Transformer block parameters
	for _, block := range m.Blocks {
		if block != nil {
			// Attention projections
			if block.Attn != nil {
				if block.Attn.QProj != nil {
					params = append(params, block.Attn.QProj.W, block.Attn.QProj.Bias)
				}
				if block.Attn.KProj != nil {
					params = append(params, block.Attn.KProj.W, block.Attn.KProj.Bias)
				}
				if block.Attn.VProj != nil {
					params = append(params, block.Attn.VProj.W, block.Attn.VProj.Bias)
				}
				if block.Attn.OutProj != nil {
					params = append(params, block.Attn.OutProj.W, block.Attn.OutProj.Bias)
				}
			}

			// LayerNorm1
			if block.Norm1 != nil {
				params = append(params, block.Norm1.Gamma, block.Norm1.Beta)
			}

			// FFN projections
			if block.FFN != nil {
				if block.FFN.Linear1 != nil {
					params = append(params, block.FFN.Linear1.W, block.FFN.Linear1.Bias)
				}
				if block.FFN.Linear2 != nil {
					params = append(params, block.FFN.Linear2.W, block.FFN.Linear2.Bias)
				}
			}

			// LayerNorm2
			if block.Norm2 != nil {
				params = append(params, block.Norm2.Gamma, block.Norm2.Beta)
			}
		}
	}

	// Final LayerNorm
	if m.FinalNorm != nil {
		params = append(params, m.FinalNorm.Gamma, m.FinalNorm.Beta)
	}

	// Output head
	if m.OutputHead != nil {
		params = append(params, m.OutputHead.W, m.OutputHead.Bias)
	}

	return params
}

// ZeroGrad clears gradients for all parameters.
func (m *LLM) ZeroGrad() {
	params := m.GetParameters()
	for _, param := range params {
		if param != nil {
			param.Grad = nil
		}
	}
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

	// Embedding table [vocabSize, embedDim] - random init with small std
	embTable, _ := be.Alloc(vocabSize * embedDim * 4)
	initWeightsNormal(embTable, vocabSize*embedDim, 0.02) // embedding std
	embShape := core.Shape{vocabSize, embedDim}
	embStrides := core.ContiguousStrides(embShape, 4)
	embTensor := tensor.New(embTable, embShape, embStrides, core.Float32)
	emb, _ := NewEmbedding(embTensor)

	// Blocks
	blocks := make([]*TransformerBlock, numLayers)
	hidden := embedDim * 4 // FFN hidden
	for L := 0; L < numLayers; L++ {
		// Q,K,V,Out projections: in=embedDim, out=embedDim
		// Use Kaiming init: std = sqrt(2 / inSize)
		allocLin := func(in, out int) (*Linear, error) {
			w, _ := be.Alloc(out * in * 4)
			// Kaiming normal: std = sqrt(2 / in)
			std := float32(math.Sqrt(2.0 / float64(in)))
			initWeightsNormal(w, out*in, std)
			b, _ := be.Alloc(out * 4)
			initWeightsZero(b, out) // bias init to zero
			wT := tensor.New(w, core.Shape{out, in}, core.ContiguousStrides(core.Shape{out, in}, 4), core.Float32)
			bT := tensor.New(b, core.Shape{out}, core.ContiguousStrides(core.Shape{out}, 4), core.Float32)
			return NewLinear(in, out, wT, bT)
		}
		q, _ := allocLin(embedDim, embedDim)
		k, _ := allocLin(embedDim, embedDim)
		v, _ := allocLin(embedDim, embedDim)
		outProj, _ := allocLin(embedDim, embedDim)
		attn := NewAttention(numHeads, headDim, q, k, v, outProj, 10000)

		// FFN: embedDim -> hidden -> embedDim, with Kaiming init
		w1, _ := be.Alloc(hidden * embedDim * 4)
		std1 := float32(math.Sqrt(2.0 / float64(embedDim)))
		initWeightsNormal(w1, hidden*embedDim, std1)
		b1, _ := be.Alloc(hidden * 4)
		initWeightsZero(b1, hidden)

		w2, _ := be.Alloc(embedDim * hidden * 4)
		std2 := float32(math.Sqrt(2.0 / float64(hidden)))
		initWeightsNormal(w2, embedDim*hidden, std2)
		b2, _ := be.Alloc(embedDim * 4)
		initWeightsZero(b2, embedDim)

		lin1, _ := NewLinear(embedDim, hidden, tensor.New(w1, core.Shape{hidden, embedDim}, core.ContiguousStrides(core.Shape{hidden, embedDim}, 4), core.Float32), tensor.New(b1, core.Shape{hidden}, core.ContiguousStrides(core.Shape{hidden}, 4), core.Float32))
		lin2, _ := NewLinear(hidden, embedDim, tensor.New(w2, core.Shape{embedDim, hidden}, core.ContiguousStrides(core.Shape{embedDim, hidden}, 4), core.Float32), tensor.New(b2, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32))
		ffn := &FeedForward{Linear1: lin1, Linear2: lin2}

		// LayerNorm: gamma init to 1, beta to 0
		gamma, _ := be.Alloc(embedDim * 4)
		initWeightsOnes(gamma, embedDim)
		beta, _ := be.Alloc(embedDim * 4)
		initWeightsZero(beta, embedDim)
		gT := tensor.New(gamma, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32)
		bT := tensor.New(beta, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32)
		n1 := NewLayerNorm(gT, bT, 1e-5)

		gamma2, _ := be.Alloc(embedDim * 4)
		initWeightsOnes(gamma2, embedDim)
		beta2, _ := be.Alloc(embedDim * 4)
		initWeightsZero(beta2, embedDim)
		n2 := NewLayerNorm(tensor.New(gamma2, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32), tensor.New(beta2, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32), 1e-5)
		blocks[L] = &TransformerBlock{Attn: attn, FFN: ffn, Norm1: n1, Norm2: n2}
	}

	// Final norm: gamma=1, beta=0
	gammaF, _ := be.Alloc(embedDim * 4)
	initWeightsOnes(gammaF, embedDim)
	betaF, _ := be.Alloc(embedDim * 4)
	initWeightsZero(betaF, embedDim)
	finalNorm := NewLayerNorm(
		tensor.New(gammaF, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32),
		tensor.New(betaF, core.Shape{embedDim}, core.ContiguousStrides(core.Shape{embedDim}, 4), core.Float32),
		1e-5)

	// Output head: embedDim -> vocabSize, Kaiming init
	wOut, _ := be.Alloc(vocabSize * embedDim * 4)
	stdOut := float32(math.Sqrt(2.0 / float64(embedDim)))
	initWeightsNormal(wOut, vocabSize*embedDim, stdOut)
	bOut, _ := be.Alloc(vocabSize * 4)
	initWeightsZero(bOut, vocabSize)

	outHead, _ := NewLinear(embedDim, vocabSize, tensor.New(wOut, core.Shape{vocabSize, embedDim}, core.ContiguousStrides(core.Shape{vocabSize, embedDim}, 4), core.Float32), tensor.New(bOut, core.Shape{vocabSize}, core.ContiguousStrides(core.Shape{vocabSize}, 4), core.Float32))
	return NewLLM(emb, blocks, finalNorm, outHead, vocabSize, embedDim, numLayers, maxSeqLen), nil
}

// initWeightsNormal fills storage with N(0, std) normal distribution
func initWeightsNormal(storage backend.Storage, n int, std float32) {
	if storage == nil || n <= 0 {
		return
	}
	b := storage.Bytes()
	if len(b) < n*4 {
		return
	}
	// Use slice casting to interpret bytes as float32
	f32Slice := make([]float32, n)
	for i := 0; i < n; i++ {
		// Box-Muller transform: convert uniform to normal
		u1 := rand.Float32()
		u2 := rand.Float32()
		for u1 < 1e-7 { // Avoid log(0)
			u1 = rand.Float32()
		}
		z0 := float32(math.Sqrt(-2.0*math.Log(float64(u1))) * math.Cos(2.0*math.Pi*float64(u2)))
		f32Slice[i] = z0 * std
	}
	// Copy float32 values to byte slice by interpreting as bytes
	ptr := unsafe.Pointer(&f32Slice[0])
	srcBytes := unsafe.Slice((*byte)(ptr), n*4)
	copy(b, srcBytes)
}

// initWeightsZero fills storage with zeros
func initWeightsZero(storage backend.Storage, n int) {
	if storage == nil || n <= 0 {
		return
	}
	b := storage.Bytes()
	if len(b) < n*4 {
		return
	}
	for i := 0; i < n*4; i++ {
		b[i] = 0
	}
}

// initWeightsOnes fills storage with ones
func initWeightsOnes(storage backend.Storage, n int) {
	if storage == nil || n <= 0 {
		return
	}
	b := storage.Bytes()
	if len(b) < n*4 {
		return
	}
	f32Slice := make([]float32, n)
	for i := 0; i < n; i++ {
		f32Slice[i] = 1.0
	}
	// Copy float32 values to byte slice
	ptr := unsafe.Pointer(&f32Slice[0])
	srcBytes := unsafe.Slice((*byte)(ptr), n*4)
	copy(b, srcBytes)
}
