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

# End Goal?
I'd like to implement a 10 bit parity with only `math` and `random` libraries