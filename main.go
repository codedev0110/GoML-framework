// Package main is a test harness for GoML: create model, forward pass, verify logits.
package main

import (
	"fmt"
	"log"

	_ "github.com/djeday123/goml/backend/cpu" // register CPU backend
	"github.com/djeday123/goml/nn"
	"github.com/djeday123/goml/tensor"
)

func main() {
	// Small config
	vocabSize := 256
	embedDim := 64
	numHeads := 4
	numLayers := 2
	maxSeqLen := 8
	batchSize := 2
	seqLen := 4

	model, err := nn.InitSmall(vocabSize, embedDim, numHeads, numLayers, maxSeqLen)
	if err != nil {
		log.Fatalf("InitSmall: %v", err)
	}

	// Dummy input: [batch, seq] int64
	indices := make([]int64, batchSize*seqLen)
	for i := range indices {
		indices[i] = int64(i % vocabSize)
	}
	input, err := tensor.FromInt64(indices, batchSize, seqLen)
	if err != nil {
		log.Fatalf("FromInt64: %v", err)
	}

	logits, err := model.Forward(input)
	if err != nil {
		log.Fatalf("Forward: %v", err)
	}

	// Verify shape: [batch, seq, vocabSize]
	if len(logits.Shape) != 3 || logits.Shape[0] != batchSize || logits.Shape[1] != seqLen || logits.Shape[2] != vocabSize {
		log.Fatalf("logits shape = %v, want [%d, %d, %d]", logits.Shape, batchSize, seqLen, vocabSize)
	}

	// Sanity: logits are finite
	f := logits.Float32()
	for i, v := range f {
		if i >= 10 {
			break
		}
		if v != v || v*v < 0 { // NaN or Inf
			log.Fatalf("logits[%d] = %f not finite", i, v)
		}
	}

	fmt.Println("OK: forward pass produced logits shape", logits.Shape)
}
