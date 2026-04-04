// Пример: Mini Transformer — классификация последовательностей
//
// Задача: определить "тональность" синтетической последовательности токенов.
// Позитивные: содержат токены {0,1,2} в начале.
// Негативные: содержат токены {3,4,5} в начале.
//
// Архитектура:
//
//	Embedding(vocabSize=8, d=32)
//	→ 2× TransformerBlock(d=32, heads=4, ffn=64)
//	→ Mean pooling по seq dim
//	→ Linear(32 → 2)
//
// Адаптировано для goml.
package main

import (
	"fmt"
	"math"
	"math/rand"
	"unsafe"

	"github.com/djeday123/goml/backend"
	_ "github.com/djeday123/goml/backend/cpu"
	"github.com/djeday123/goml/core"
	"github.com/djeday123/goml/nn"
	"github.com/djeday123/goml/tensor"
)

func floatSlice(s backend.Storage, n int) []float32 {
	if n == 0 {
		return nil
	}
	b := s.Bytes()
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), n)
}

func initWeightsNormal(storage backend.Storage, n int, std float32) {
	data := floatSlice(storage, n)
	for i := 0; i < n; i++ {
		data[i] = float32(rand.NormFloat64()) * std
	}
}

func initWeightsZero(storage backend.Storage, n int) {
	data := floatSlice(storage, n)
	for i := 0; i < n; i++ {
		data[i] = 0
	}
}

const (
	tfVocab  = 8
	tfDim    = 32
	tfHeads  = 4
	tfFFN    = 64
	tfSeqLen = 10
	tfTrain  = 500
	tfTest   = 100
	tfEpochs = 30
	tfLR     = 0.001
	tfBatch  = 16
)

// TransformerClassifier: Embedding + 2 Transformer layers + mean pool + Linear.
type TransformerClassifier struct {
	emb    *nn.Embedding
	layer1 *nn.TransformerBlock
	layer2 *nn.TransformerBlock
	fc     *nn.Linear
}

// NewTransformerClassifier creates the model.
func NewTransformerClassifier() (*TransformerClassifier, error) {
	be, err := backend.GetForDevice(backend.CPU0)
	if err != nil {
		return nil, err
	}
	headDim := tfDim / tfHeads

	// Embedding table [tfVocab, tfDim] - random init
	embTable, _ := be.Alloc(tfVocab * tfDim * 4)
	initWeightsNormal(embTable, tfVocab*tfDim, 0.02)
	embShape := core.Shape{tfVocab, tfDim}
	embStrides := core.ContiguousStrides(embShape, 4)
	embTensor := tensor.New(embTable, embShape, embStrides, core.Float32)
	emb, _ := nn.NewEmbedding(embTensor)

	// Helper to alloc Linear
	allocLin := func(in, out int) (*nn.Linear, error) {
		w, _ := be.Alloc(out * in * 4)
		std := float32(math.Sqrt(2.0 / float64(in)))
		initWeightsNormal(w, out*in, std)
		b, _ := be.Alloc(out * 4)
		initWeightsZero(b, out)
		wT := tensor.New(w, core.Shape{out, in}, core.ContiguousStrides(core.Shape{out, in}, 4), core.Float32)
		bT := tensor.New(b, core.Shape{out}, core.ContiguousStrides(core.Shape{out}, 4), core.Float32)
		return nn.NewLinear(in, out, wT, bT)
	}

	// Layer1
	q1, _ := allocLin(tfDim, tfDim)
	k1, _ := allocLin(tfDim, tfDim)
	v1, _ := allocLin(tfDim, tfDim)
	o1, _ := allocLin(tfDim, tfDim)
	attn1 := nn.NewAttention(tfHeads, headDim, q1, k1, v1, o1, 10000)

	// FFN1
	w11, _ := be.Alloc(tfFFN * tfDim * 4)
	initWeightsNormal(w11, tfFFN*tfDim, float32(math.Sqrt(2.0/float64(tfDim))))
	b11, _ := be.Alloc(tfFFN * 4)
	initWeightsZero(b11, tfFFN)
	w1T := tensor.New(w11, core.Shape{tfFFN, tfDim}, core.ContiguousStrides(core.Shape{tfFFN, tfDim}, 4), core.Float32)
	b1T := tensor.New(b11, core.Shape{tfFFN}, core.ContiguousStrides(core.Shape{tfFFN}, 4), core.Float32)
	l11, _ := nn.NewLinear(tfDim, tfFFN, w1T, b1T)
	w12, _ := be.Alloc(tfDim * tfFFN * 4)
	initWeightsNormal(w12, tfDim*tfFFN, float32(math.Sqrt(2.0/float64(tfFFN))))
	b12, _ := be.Alloc(tfDim * 4)
	initWeightsZero(b12, tfDim)
	w2T := tensor.New(w12, core.Shape{tfDim, tfFFN}, core.ContiguousStrides(core.Shape{tfDim, tfFFN}, 4), core.Float32)
	b2T := tensor.New(b12, core.Shape{tfDim}, core.ContiguousStrides(core.Shape{tfDim}, 4), core.Float32)
	l12, _ := nn.NewLinear(tfFFN, tfDim, w2T, b2T)
	ffn1 := &nn.FeedForward{Linear1: l11, Linear2: l12, SwiGLU: false}

	// Norms1
	gamma1Data := make([]float32, tfDim)
	for i := range gamma1Data {
		gamma1Data[i] = 1
	}
	gamma1, _ := tensor.FromFloat32(gamma1Data, tfDim)
	beta1Data := make([]float32, tfDim)
	beta1, _ := tensor.FromFloat32(beta1Data, tfDim)
	norm11 := nn.NewLayerNorm(gamma1, beta1, 1e-5)
	gamma2, _ := tensor.FromFloat32(gamma1Data, tfDim)
	beta2, _ := tensor.FromFloat32(beta1Data, tfDim)
	norm12 := nn.NewLayerNorm(gamma2, beta2, 1e-5)

	layer1 := &nn.TransformerBlock{Attn: attn1, FFN: ffn1, Norm1: norm11, Norm2: norm12}

	// Layer2 (similar)
	q2, _ := allocLin(tfDim, tfDim)
	k2, _ := allocLin(tfDim, tfDim)
	v2, _ := allocLin(tfDim, tfDim)
	o2, _ := allocLin(tfDim, tfDim)
	attn2 := nn.NewAttention(tfHeads, headDim, q2, k2, v2, o2, 10000)

	w21, _ := be.Alloc(tfFFN * tfDim * 4)
	initWeightsNormal(w21, tfFFN*tfDim, float32(math.Sqrt(2.0/float64(tfDim))))
	b21, _ := be.Alloc(tfFFN * 4)
	initWeightsZero(b21, tfFFN)
	w21T := tensor.New(w21, core.Shape{tfFFN, tfDim}, core.ContiguousStrides(core.Shape{tfFFN, tfDim}, 4), core.Float32)
	b21T := tensor.New(b21, core.Shape{tfFFN}, core.ContiguousStrides(core.Shape{tfFFN}, 4), core.Float32)
	l21, _ := nn.NewLinear(tfDim, tfFFN, w21T, b21T)
	w22, _ := be.Alloc(tfDim * tfFFN * 4)
	initWeightsNormal(w22, tfDim*tfFFN, float32(math.Sqrt(2.0/float64(tfFFN))))
	b22, _ := be.Alloc(tfDim * 4)
	initWeightsZero(b22, tfDim)
	w22T := tensor.New(w22, core.Shape{tfDim, tfFFN}, core.ContiguousStrides(core.Shape{tfDim, tfFFN}, 4), core.Float32)
	b22T := tensor.New(b22, core.Shape{tfDim}, core.ContiguousStrides(core.Shape{tfDim}, 4), core.Float32)
	l22, _ := nn.NewLinear(tfFFN, tfDim, w22T, b22T)
	ffn2 := &nn.FeedForward{Linear1: l21, Linear2: l22, SwiGLU: false}

	gamma3, _ := tensor.FromFloat32(gamma1Data, tfDim)
	beta3, _ := tensor.FromFloat32(beta1Data, tfDim)
	norm21 := nn.NewLayerNorm(gamma3, beta3, 1e-5)
	gamma4, _ := tensor.FromFloat32(gamma1Data, tfDim)
	beta4, _ := tensor.FromFloat32(beta1Data, tfDim)
	norm22 := nn.NewLayerNorm(gamma4, beta4, 1e-5)

	layer2 := &nn.TransformerBlock{Attn: attn2, FFN: ffn2, Norm1: norm21, Norm2: norm22}

	// FC
	fcW, _ := be.Alloc(2 * tfDim * 4)
	initWeightsNormal(fcW, 2*tfDim, float32(math.Sqrt(2.0/float64(tfDim))))
	fcB, _ := be.Alloc(2 * 4)
	initWeightsZero(fcB, 2)
	fcWT := tensor.New(fcW, core.Shape{2, tfDim}, core.ContiguousStrides(core.Shape{2, tfDim}, 4), core.Float32)
	fcBT := tensor.New(fcB, core.Shape{2}, core.ContiguousStrides(core.Shape{2}, 4), core.Float32)
	fc, _ := nn.NewLinear(tfDim, 2, fcWT, fcBT)

	return &TransformerClassifier{
		emb:    emb,
		layer1: layer1,
		layer2: layer2,
		fc:     fc,
	}, nil
}

// Parameters collects all trainable parameters.
func (m *TransformerClassifier) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	params = append(params, m.emb.Table)
	// Layer1
	params = append(params, m.layer1.Attn.QProj.W, m.layer1.Attn.QProj.Bias)
	params = append(params, m.layer1.Attn.KProj.W, m.layer1.Attn.KProj.Bias)
	params = append(params, m.layer1.Attn.VProj.W, m.layer1.Attn.VProj.Bias)
	params = append(params, m.layer1.Attn.OutProj.W, m.layer1.Attn.OutProj.Bias)
	params = append(params, m.layer1.FFN.Linear1.W, m.layer1.FFN.Linear1.Bias)
	params = append(params, m.layer1.FFN.Linear2.W, m.layer1.FFN.Linear2.Bias)
	params = append(params, m.layer1.Norm1.Gamma, m.layer1.Norm1.Beta)
	params = append(params, m.layer1.Norm2.Gamma, m.layer1.Norm2.Beta)
	// Layer2
	params = append(params, m.layer2.Attn.QProj.W, m.layer2.Attn.QProj.Bias)
	params = append(params, m.layer2.Attn.KProj.W, m.layer2.Attn.KProj.Bias)
	params = append(params, m.layer2.Attn.VProj.W, m.layer2.Attn.VProj.Bias)
	params = append(params, m.layer2.Attn.OutProj.W, m.layer2.Attn.OutProj.Bias)
	params = append(params, m.layer2.FFN.Linear1.W, m.layer2.FFN.Linear1.Bias)
	params = append(params, m.layer2.FFN.Linear2.W, m.layer2.FFN.Linear2.Bias)
	params = append(params, m.layer2.Norm1.Gamma, m.layer2.Norm1.Beta)
	params = append(params, m.layer2.Norm2.Gamma, m.layer2.Norm2.Beta)
	// FC
	params = append(params, m.fc.W, m.fc.Bias)
	return params
}

// Forward: tokens []int → logits [1, 2]
func (m *TransformerClassifier) Forward(tokens []int64) (*tensor.Tensor, error) {
	// Convert tokens to tensor [seqLen]
	tokenTensor, err := tensor.FromInt64(tokens, tfSeqLen)
	if err != nil {
		return nil, err
	}

	// Embedding [seqLen, d]
	embOut, err := m.emb.Forward(tokenTensor)
	if err != nil {
		return nil, err
	}

	// Add batch dim: [1, seqLen, d]
	newShape := core.Shape{1, tfSeqLen, tfDim}
	newStrides := core.ContiguousStrides(newShape, 4)
	embOut = tensor.New(embOut.Storage, newShape, newStrides, core.Float32)

	// 2 Transformer layers
	h, err := m.layer1.Forward(embOut)
	if err != nil {
		return nil, err
	}
	h, err = m.layer2.Forward(h)
	if err != nil {
		return nil, err
	}

	// Mean pooling по seq dim: [1, d]
	pooled, err := meanPool(h, tfSeqLen, tfDim)
	if err != nil {
		return nil, err
	}

	// Классификатор: [1, d] → [1, 2]
	return m.fc.Forward(pooled)
}

// meanPool: усредняет [1, seqLen, d] → [1, d].
func meanPool(x *tensor.Tensor, seqLen, d int) (*tensor.Tensor, error) {
	data := floatSlice(x.Storage, x.NumElements())
	outData := make([]float32, d)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < d; j++ {
			outData[j] += data[i*d+j]
		}
	}
	for j := range outData {
		outData[j] /= float32(seqLen)
	}
	be, err := backend.GetForDevice(x.Storage.Device())
	if err != nil {
		return nil, err
	}
	outStorage, err := be.Alloc(d * 4)
	if err != nil {
		return nil, err
	}
	outFloat := floatSlice(outStorage, d)
	copy(outFloat, outData)
	outShape := core.Shape{1, d}
	outStrides := core.ContiguousStrides(outShape, 4)
	out := tensor.New(outStorage, outShape, outStrides, core.Float32)
	return out, nil
}

// generateData: создаёт (tokens, label) пары.
// label=0 (негативный): начало с {3,4,5}
// label=1 (позитивный): начало с {0,1,2}
func generateData(n int) ([][]int64, []int64) {
	seqs := make([][]int64, n)
	labels := make([]int64, n)
	for i := 0; i < n; i++ {
		label := int64(rand.Intn(2))
		labels[i] = label
		seq := make([]int64, tfSeqLen)
		for j := range seq {
			if j < 3 {
				if label == 1 {
					seq[j] = int64(rand.Intn(3)) // {0,1,2}
				} else {
					seq[j] = 3 + int64(rand.Intn(3)) // {3,4,5}
				}
			} else {
				seq[j] = int64(rand.Intn(tfVocab))
			}
		}
		seqs[i] = seq
	}
	return seqs, labels
}

func main() {
	rand.Seed(42)

	trainSeqs, trainLabels := generateData(tfTrain)
	testSeqs, testLabels := generateData(tfTest)

	fmt.Printf("Vocab: %d, seqLen: %d, d_model: %d, heads: %d\n",
		tfVocab, tfSeqLen, tfDim, tfHeads)
	fmt.Printf("Train: %d, Test: %d\n\n", tfTrain, tfTest)

	model, err := NewTransformerClassifier()
	if err != nil {
		panic(err)
	}

	// opt := optim.NewAdamW(model.Parameters(), tfLR, 0.9, 0.999, 1e-8, 0.01) // removed, manual update

	totalP := 0
	for _, p := range model.Parameters() {
		totalP += p.NumElements()
	}
	fmt.Printf("Параметров: %d\n\n", totalP)
	fmt.Printf("%-6s  %-10s  %-12s  %-10s\n", "Epoch", "Loss", "Train Acc", "Test Acc")
	fmt.Println("──────────────────────────────────────────────")

	for epoch := 1; epoch <= tfEpochs; epoch++ {
		perm := rand.Perm(tfTrain)
		totalLoss := 0.0
		steps := 0

		for b := 0; b < tfTrain; b += tfBatch {
			end := b + tfBatch
			if end > tfTrain {
				end = tfTrain
			}

			batchLoss := 0.0
			// Zero grads
			for _, p := range model.Parameters() {
				if p.Grad != nil {
					p.Grad = nil
				}
			}

			for _, idx := range perm[b:end] {
				logits, err := model.Forward(trainSeqs[idx])
				if err != nil {
					panic(err)
				}
				// Manual loss and grad
				logitsF := logits.Float32() // [2]
				maxV := logitsF[0]
				if logitsF[1] > maxV {
					maxV = logitsF[1]
				}
				exp0 := math.Exp(float64(logitsF[0] - maxV))
				exp1 := math.Exp(float64(logitsF[1] - maxV))
				sumExp := exp0 + exp1
				soft0 := exp0 / sumExp
				soft1 := exp1 / sumExp
				grad0 := soft0
				grad1 := soft1
				target := trainLabels[idx]
				if target == 0 {
					grad0 -= 1
				} else {
					grad1 -= 1
				}
				// Loss
				lossVal := 0.0
				if target == 0 {
					lossVal = -math.Log(soft0)
				} else {
					lossVal = -math.Log(soft1)
				}
				batchLoss += lossVal

				// Update fc
				// Recompute pooled
				tokenTensor, _ := tensor.FromInt64(trainSeqs[idx], tfSeqLen)
				embOut, _ := model.emb.Forward(tokenTensor)
				newShape := core.Shape{1, tfSeqLen, tfDim}
				newStrides := core.ContiguousStrides(newShape, 4)
				embOut = tensor.New(embOut.Storage, newShape, newStrides, core.Float32)
				h, _ := model.layer1.Forward(embOut)
				h, _ = model.layer2.Forward(h)
				pooled, _ := meanPool(h, tfSeqLen, tfDim)
				pooledF := pooled.Float32()     // [tfDim]
				fcWF := model.fc.W.Float32()    // [2, tfDim]
				fcBF := model.fc.Bias.Float32() // [2]
				learningRate := float32(0.01)
				for j := 0; j < tfDim; j++ {
					fcWF[0*tfDim+j] -= learningRate * float32(grad0) * pooledF[j]
					fcWF[1*tfDim+j] -= learningRate * float32(grad1) * pooledF[j]
				}
				fcBF[0] -= learningRate * float32(grad0)
				fcBF[1] -= learningRate * float32(grad1)
			}
			// Average loss
			batchLoss /= float64(end - b)
			// Manual update above, no opt.Step()

			totalLoss += batchLoss
			steps++
		}

		avgLoss := totalLoss / float64(steps)

		if epoch%5 == 0 || epoch == 1 {
			trainAcc := evalTransformer(model, trainSeqs[:100], trainLabels[:100])
			testAcc := evalTransformer(model, testSeqs, testLabels)
			fmt.Printf("%-6d  %-10.4f  %-12.2f%%  %-10.2f%%\n",
				epoch, avgLoss, trainAcc*100, testAcc*100)
		}
	}

	fmt.Println("\n✅ Transformer обучен!")

	// Пример предсказания
	fmt.Println("\nПримеры предсказаний:")
	for i := 0; i < 5; i++ {
		logits, err := model.Forward(testSeqs[i])
		if err != nil {
			panic(err)
		}
		logitsF := logits.Float32()
		class := 0
		if logitsF[1] > logitsF[0] {
			class = 1
		}
		correct := "✓"
		if int64(class) != testLabels[i] {
			correct = "✗"
		}
		fmt.Printf("  Seq: %v → pred=%d (true=%d) %s\n",
			testSeqs[i][:5], class, testLabels[i], correct)
	}
}

func evalTransformer(model *TransformerClassifier, seqs [][]int64, labels []int64) float64 {
	correct := 0
	for i, seq := range seqs {
		logits, err := model.Forward(seq)
		if err != nil {
			continue
		}
		logitsF := logits.Float32()
		pred := 0
		if logitsF[1] > logitsF[0] {
			pred = 1
		}
		if pred == int(labels[i]) {
			correct++
		}
	}
	return float64(correct) / float64(len(labels))
}
