# v3: Don't neglect your children!
I implemented backpropogration, so the machine does learn, no promises on how fast though. This is where I started realising maybe a neuron based abstraction wasn't the real answer to solve this. Python has a lot of object overhead which blocks your training speed.

There is a subtle flaw in this though. Everytime a child updates the upstream/parent neuron, I immediately changed it's weights and biases... which means the next child who talks to the neuron is talking to a *different* parent.
