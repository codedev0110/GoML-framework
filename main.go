// Package main — GoML: mətndən giriş alır, model forward edir, logits və nümunə çıxış verir.
package main

import (
	"fmt"
	"log"
	"os"

	_ "github.com/djeday123/goml/backend/cpu"
	"github.com/djeday123/goml/nn"
	"github.com/djeday123/goml/tensor"
	"github.com/djeday123/goml/tokenizer"
)

func main() {
	// Giriş: əgər arqument varsa onu götür, yoxdursa default mətn
	inputText := "hello"
	if len(os.Args) > 1 {
		inputText = os.Args[1]
	}

	vocabSize := 256
	embedDim := 64
	numHeads := 4
	numLayers := 2
	maxSeqLen := 64

	model, err := nn.InitSmall(vocabSize, embedDim, numHeads, numLayers, maxSeqLen)
	if err != nil {
		log.Fatalf("Model yüklənmədi: %v", err)
	}

	tok := tokenizer.NewByteLevel()
	ids := tok.Encode(inputText)
	if len(ids) == 0 {
		ids = []int64{0}
	}
	if len(ids) > maxSeqLen {
		ids = ids[:maxSeqLen]
	}

	// Tensor: [1, seqLen] — bir batch, bir sətir
	seqLen := len(ids)
	input, err := tensor.FromInt64(ids, 1, seqLen)
	if err != nil {
		log.Fatalf("Input tensor: %v", err)
	}

	logits, err := model.Forward(input)
	if err != nil {
		log.Fatalf("Forward: %v", err)
	}

	// Çıxış
	fmt.Println("--- Giriş ---")
	fmt.Println("Mətn:", inputText)
	fmt.Println("Token ID-lər:", ids)
	fmt.Println()
	fmt.Println("--- Model çıxışı ---")
	fmt.Println("Logits shape:", logits.Shape) // [1, seqLen, 256]

	// Son pozisiyada ən çox ehtimal olunan token (sadə “növbəti simvol” proqnozu)
	logitsF := logits.Float32()
	lastPos := (seqLen - 1) * vocabSize
	bestIdx := 0
	for i := 1; i < vocabSize; i++ {
		if logitsF[lastPos+i] > logitsF[lastPos+bestIdx] {
			bestIdx = i
		}
	}
	nextToken := tok.Decode([]int64{int64(bestIdx)})
	fmt.Println("Növbəti token (argmax):", bestIdx, "→", nextToken)
	fmt.Println()
	fmt.Println("OK.")
}
