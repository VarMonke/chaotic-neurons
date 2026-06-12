# v5: Neurons are a hoax?
`v4` works. It's a proper feedforward network written in pure python. A few issues though:
1. Python has a lot of object overhead.
2. numpy is very good for doing matrix things(it's amazing for a lot of other things too but hey, for this case, matrix multiplications are blazing fast when you use numpy)

This is the last version of `/primitive`. I've realised some important things, I had an intuition for RNNs, and then found out why RNNs fail, then LSTMs etc etc. I think this is enough first princples to treat myself to a little bit of `import torch`.