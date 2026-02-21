package train

import (
	"github.com/djeday123/goml/nn"
	"github.com/djeday123/goml/tensor"
)

// Trainer runs the training loop: forward, loss, backward, step.
type Trainer struct {
	Model     *nn.LLM
	Optimizer nn.Optimizer
}

// NewTrainer creates a trainer.
func NewTrainer(model *nn.LLM, opt nn.Optimizer) *Trainer {
	return &Trainer{Model: model, Optimizer: opt}
}

// Step runs one step: forward on inputs, loss with targets, backward (placeholder), optimizer step.
func (t *Trainer) Step(inputs, targets *tensor.Tensor) (loss float32, err error) {
	logits, err := t.Model.Forward(inputs)
	if err != nil {
		return 0, err
	}
	// Flatten for loss: logits [batch, seq, C], targets [batch, seq]
	batch, seq := logits.Shape[0], logits.Shape[1]
	C := logits.Shape[2]
	logitsFlat, err := logits.View(batch*seq, C)
	if err != nil {
		return 0, err
	}
	targetsFlat, err := targets.View(batch * seq)
	if err != nil {
		return 0, err
	}
	lossTensor, err := nn.CrossEntropyLoss(logitsFlat, targetsFlat)
	if err != nil {
		return 0, err
	}
	loss = lossTensor.Float32()[0]
	nn.Backward(lossTensor)
	t.Optimizer.Step()
	return loss, nil
}
