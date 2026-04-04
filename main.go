// Package main — Sadə demo: train + inference bir ərəfədə
package main

import (
	"fmt"
	"log"
	"math"
	"os"

	_ "github.com/djeday123/goml/backend/cpu"
	"github.com/djeday123/goml/nn"
	"github.com/djeday123/goml/optim"
	"github.com/djeday123/goml/tensor"
	"github.com/djeday123/goml/tokenizer"
)

func main() {
	text := "hi"
	if len(os.Args) > 1 {
		text = os.Args[1]
	}
	vocabSize := 256
	embedDim := 64
	numHeads := 4
	numLayers := 2
	epochs := 20

	// Model
	model, err := nn.InitSmall(vocabSize, embedDim, numHeads, numLayers, 64)
	if err != nil {
		log.Fatal(err)
	}

	tok := tokenizer.NewByteLevel()
	tokenIDs := tok.Encode(text)
	if len(tokenIDs) == 0 {
		tokenIDs = []int64{0}
	}

	seqLen := len(tokenIDs)
	input, _ := tensor.FromInt64(tokenIDs, 1, seqLen)

	// Target: next token
	target := make([]int64, seqLen)
	for i := 0; i < seqLen-1; i++ {
		target[i] = tokenIDs[i+1]
	}
	target[seqLen-1] = tokenIDs[seqLen-1]
	targetTensor, _ := tensor.FromInt64(target, 1, seqLen)

	fmt.Printf("📝 Text: %q → [%v]\n", text, tokenIDs)
	fmt.Printf("🎯 Target: [%v]\n\n", target)

	// Train
	fmt.Println("⚙️  Training...")
	learningRate := float32(0.01)
	targetF := targetTensor.Int64()

	// Optimizer with all model parameters
	params := model.GetParameters()
	_ = optim.NewAdamW(params, 0.001, 0.9, 0.999, 1e-8, 0.01) // Would be used, but we'll use simple SGD for now

	for epoch := 0; epoch < epochs; epoch++ {
		// Zero gradients
		for _, p := range params {
			if p != nil {
				p.Grad = nil
			}
		}

		logits, _ := model.Forward(input)
		logitsF := logits.Float32()

		// Compute loss and gradients
		loss := float32(0.0)
		logitsGrad := make([]float32, len(logitsF))

		for i := 0; i < seqLen; i++ {
			row := logitsF[i*vocabSize : (i+1)*vocabSize]
			t := int(targetF[i])

			// Softmax with stability
			maxVal := row[0]
			for j := 1; j < vocabSize; j++ {
				if row[j] > maxVal {
					maxVal = row[j]
				}
			}

			sumExp := float32(0)
			expVals := make([]float32, vocabSize)
			for j := 0; j < vocabSize; j++ {
				expVals[j] = float32(math.Exp(float64(row[j] - maxVal)))
				sumExp += expVals[j]
			}

			// Loss
			loss -= (row[t] - maxVal - float32(math.Log(float64(sumExp))))

			// Gradient: softmax - one_hot
			for j := 0; j < vocabSize; j++ {
				soft := expVals[j] / sumExp
				logitsGrad[i*vocabSize+j] = soft
			}
			logitsGrad[i*vocabSize+t] -= 1.0
		}
		loss /= float32(seqLen)

		// DIRECTLY UPDATE OUTPUT HEAD (since backward pass isn't fully wired through)
		// y = x @ W^T + b
		// We have: dy/d(bias) = 1
		// So: update bias by -lr * sum(logitsGrad) for each output

		outHeadBias := model.OutputHead.Bias
		if outHeadBias != nil {
			biasF := outHeadBias.Float32()
			for j := 0; j < vocabSize; j++ {
				// Sum gradient for this output index across all positions
				gradSum := float32(0)
				for i := 0; i < seqLen; i++ {
					gradSum += logitsGrad[i*vocabSize+j]
				}
				gradSum /= float32(seqLen)
				biasF[j] -= learningRate * gradSum
			}
		}

		fmt.Printf("  Epoch %d: Loss=%.4f\n", epoch, loss)
	}

	// Test
	fmt.Println("\n✅ Inference:")
	logits, _ := model.Forward(input)
	logitsF := logits.Float32()

	for i := 0; i < seqLen; i++ {
		row := logitsF[i*vocabSize : (i+1)*vocabSize]
		maxIdx := 0
		for j := 1; j < vocabSize; j++ {
			if row[j] > row[maxIdx] {
				maxIdx = j
			}
		}
		t := int(targetF[i])
		status := "✗"
		if maxIdx == t {
			status = "✓"
		}
		fmt.Printf("  Pos %d: expected=%d, predicted=%d %s\n", i, t, maxIdx, status)
	}
	fmt.Println("\n✨ Done!")
}
