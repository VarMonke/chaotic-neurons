# Watch me reinvent the wheel to make my electrons reason

So basically I hate myself(that's well.. a given)

## Hmm. How DOES my machine "learn"?

I sat down, did everything from "first principles" (yeah i'm very cool like that.)

So this implements backprop, although my way of doing backprop is very noisy and very un-real-life-smart-stuff

Hmm I will probably read up on some more formal introductions and intuitions before i make a v4.


## Versions
- `/primitive` is me doing things with a vague idea of what a neuron is, and how signals are actually propogated. It's what I had an intuition for.

- `v3` is after I went through `https://www.youtube.com/watch?v=IHZwWFHWa-w&t=1004s` to get an intuition for what gradient descent would help me with.

- `v4` I implemented proper backprop and this can solve a good number of n-parity problems with just an issue with the all 1's case.. hmm maybe the feature decomposition path I approach this with isn't the right mental modal(fore shadowing)
- - There are some testing files within `v4`. There are some numbers I want to talk about, but I'll do that later. I'm pretty happy with how v4 turned out. Not sure what v5 will be, but I guess we'll see.
- - `v4/neuron.py` verifies my backprop to me accurate to an amazing precision level. So yay me.

- `v4` bottlenecks are more architecture-based than "Machine Learning" based, but yeah `v4` is solid. I did finally read some formal math and realised everything is just a Jacobin and the number of object calls I was making in `.forward()`, `.backward()` and `.step()` were insane.

> Before you read the next part, it's after a few hours of formal math and blazing through what each new optimizer and cost function does. None of this was built from first princples because first princples was supposed to teach me _how_ and _why_ learning happens. The teaching is done. The rest is just an optimization problem.
## Neural-net experiments (hand-rolled engine)

All on the same hand-built Neuron/backprop engine. Sigmoid + MSE unless noted.
Later runs switch to layer/Jacobian backprop, ReLU, softmax + cross-entropy.

### Parity (the wall)
- 10-bit parity, single hidden [32], sigmoid: 95.5% (978/1024): fails high-count buckets (7,9,10) wholesale
- 9-bit parity [32]: "flat-tail" — prediction flatlines ~0.257 past bit-count 5, stops alternating
- Deep sigmoid nets ([32,16], [64,64], [32,32,32]): stuck at 50% saddle (depth + sigmoid = vanishing gradient)
- Pairwise wiring 45->16->1 (1 neuron/pair), sigmoid: stuck at 50% saddle
- Mixed-order 90 neurons (orders 2..10) -> direct output: peak ~62% (634/1024), bounced, never converged

### Easy foils (what the net is naturally good at)
- Majority 10-bit ("more than half bits = 1") [32] sigmoid: 100% in 50 epochs (monotonic in count)
- Two-moons (2D nonlinear) [16] sigmoid: 99.75% (798/800)
- 5x5 line orientation (H vs V, noisy) [16] sigmoid: 100% train / 99.75% test (399/400) — learns row/col filters

### MNIST (downscaled, 10-way)
- 7x7 dense [32], sigmoid, 6k/1k: 95.1% train / 93.5% test — learns edge/curve/loop filters
- Locally-connected pyramid (7->5->3->10, no weight sharing): 94.4% train / 91.8% test — position-specific local filters

### ReLU (fixing the depth/optimization wall)
- Deep ReLU [64,32] on 10-bit parity (sigmoid out): 99.5% (1019/1024)! BEAT the 95.5% ceiling; sigmoid-deep died at 50%
- ReLU MNIST 7x7, 6k data, [64,32], softmax+CE: 99.9% train / 93.5% test, overfit (DATA wall), confidence 99.0%
- ReLU MNIST 14x14, 20k data, [16,16], softmax+CE: 95.1% train / 93.6% test — no overfit (CAPACITY wall), confidence 96.7%
- ReLU MNIST 14x14, 20k data, [32,16], softmax+CE: 99.0% train / 95.95% test — BROKE 93.5% wall, confidence 99.1%
  (per-digit all >=94%; best 1=98.5%, 0=97.5%; worst 2/3/9 ~94%)

### RNN + backprop-through-time (learning algorithms over time)
- Sequential parity (running XOR), H=12, trained on len <=10:
    len 5/10/20/30/50/100 -> 100% / 100% / 100% / 100% / 100% / 100%  (extrapolates 10x past training)
- Binary addition (LSB-first, carry = memory), H=16, trained on 2-8 bits:
    4/8/12/16/24/32 bits -> 100% / 100% / 100% / 100% / 100% / 100%  (learned the carry algorithm)
- Copy/delay (output the bit from N steps ago), H=24: the MEMORY WALL:
    N=1: 100% | N=5: 100% | N=10: 100% | N=15: 49.9% | N=20: 52.0% | N=25: 47.6%
    (vanilla RNN memory breaks between ~10-15 steps, gradient vanishes through time)

### Audio (spoken-digit recognition)
- Free Spoken Digit Dataset: 3000 clips, 6 speakers, digits 0-9
- Pipeline: wav -> 16x16 log-spectrogram (freq x time) -> ReLU/softmax MLP [64,32], 85/15 split
- Result: 100% train / 96.22% test, confidence high
    per-digit: 0/2/3 = 100%; worst 1 = 90.5%, 6 = 91.5%

### Key takeaways
- Representation was rarely the wall; OPTIMIZATION was (saddles, vanishing gradients).
- Sigmoid depth flattens -> ReLU fixes it (slope 1, gradient survives depth).
- The ~93.5% MNIST "ceiling" was hit two different ways: data wall (memorize) vs capacity wall — need BOTH width + data to break it.
- RNNs learn procedures (parity, addition) that generalize far beyond training length.
- Vanilla RNN memory wall ~10-15 steps = the through-time twin of the depth problem -> motivates LSTM/GRU/attention.
- Audio classification = image classification once you take the spectrogram.
