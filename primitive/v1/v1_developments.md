# v1: Can I brute force XOR to work?
`v1` had very little improvements over `v0`. To a human(yes I'm a human too), we'd say XOR = OR AND (NOT AND). That's a very valid `feature decomposition` for us, because the abstraction makes sense.

Which is what I've realised in `v1`. Funnily enough when I implement a Network with hidden layers, the machine finds another abstraction which it deems to be "enough" to understand XOR. I wouldn't have come up with that, but hey, as long as the machine learns right?