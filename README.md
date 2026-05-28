# Watch me reinvent the wheel to make my electrons reason

So basically I hate myself(that's well.. a given)

## Hmm. How DOES my machine "learn"?

I sat down, did everything from "first principles" (yeah i'm very cool like that.)

So this implements backprop, although my way of doing backprop is very noisy and very un-real-life-smart-stuff

Hmm I will probably read up on some more formal introductions and intuitions before i make a v4.


## Versions
- `/primitive` is me doing things with a vague idea of what a neuron is, and how signals are actually propogated. It's what I had an intuition for.

- `v3` is after I went through `https://www.youtube.com/watch?v=IHZwWFHWa-w&t=1004s` to get an intuition for what gradient descent would help me with.

- `v4`(soon) will be after I read a bit more formal things because my signal propogation, the weight initialization are probably not the best way to do things. I realised I update a neuron everytime it's called from a child, but the gradients should accumulate to show the "net effect I have on the child neurons dependant on me" to the neuron, and to make it figure itself out

# End Goal?
I'd like to implement a 10 bit parity with only `math` and `random` libraries