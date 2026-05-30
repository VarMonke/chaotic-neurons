# How does an electron, which can jump between energy states, reason?

The simple answer is: __it can't__. Not on its own.

So the natural next question is:

Can a group of electrons coordinate together to behave intelligently?

Again, the simple answer is __no__.

But a group of electrons can collectively represent information. They can occupy different electrical states, move through materials, and respond to electric fields in predictable ways.

The challenge then becomes finding a way to interpret these collective states. So you build some scaffolding for these electrons' messy states to interact coherently, which is a MOSFET.
Now why is it called a MOSFET, why did engineers end up designing a MOSFET like system is something I don't really care about, because I don't particularly enjoy electronics.
But the real answer is, I'm somewhat lazy. I could theoretically sit down and go into the rabbit hole of, "Why an electron is being asked to reason intelligently?". There are many answers to it.
But these answers don't change my implementation.

So a MOSFET is not a reasoning machine. It is a decision rule embedded in physics. Given a particular gate voltage, it determines whether current may flow or not. In other words, it converts the chaotic motion of countless electrons into a simple opinion:
ON or OFF.
One MOSFET is not intelligent. It is merely a physical implementation of a tiny logical decision.

But then again, we thought a single electron is stupid, and so are a group of electrons being hearded by this MOSFET. So our next logical progression is... what does a group of MOSFETs tell you? Do they behave intelligently?

The simple answer, surprisingly is: __somewhat__.

No they can't figure out how to solve n-bit parity yet, but they've moved from a dumb electron to a logic gated "system".
Huh. Notice how everything is a system. MOSFETs are organized into a system, which in turn is a system of electrons, and an electron is... whatever an electron is. I don't really care.

Logic emerges. Then there's some mix and match of Logic to give you Memory.
Then there is some more mix and matching of Memory to give you Computation.
Again, I should care about these, but for this scope of making my electrons "reason", I don't really care.

Huh. So a recursive system of dumb things _can_ reason. Huh.
And eventually, what we recognize as "reasoning" emerges.

Then again, what is "reasoning"? Reasoning is just a way we convince ourself things happen. It's natural to us. It might be absurd to dumb electron.
No electron understands the problem.
No MOSFET understands the program.
Yet together they form a system capable of understanding both.
H. u. h.

## Moving away from electronics for a while
So a dumb simplistic unit can help in reasoning by using multiple of itself, and organising itself in some specific way. Okay. What is reasoning?
I'd like to think of reasoning as a way you do `output = f(input)`.
`f` could be simple addition. Ooo. Multiplication? But that isn't what helps you reason.

To keep it simple, let us say you're trying to "reason" what a cat is. (or dogs, but I like cats.)
I show you 500 pictures of _something_ and I assure you, "Hey. This is a cat, you can trust me".
If I tell you, "Go through these 500, and then I'll show you a picutre of something, if you can confidently tell me it's a cat, I'll spare you."
This breeds the question, what is "confidently"? Again, confidence can also be decomposed into a system.

Somehow, you don't need to look at 500 pictures to "reason" a cat. You might understand with only 10 pictures. But then again, your input scope might also be limited. Which is another system.

So let's say you sit down and look at all 500 examples, you probably could tell me that.. somewhere along the way of "reasoning" cats, you've build a mental model for what a cat is.

Whiskers. Face Shape. Pointy Ears. Paws.

Well these are ways _you_ reason, and well these are coherent to _us_. Sadly an electron doesn't really care about cats as much as you and I. Infact, an electron itself doesn't know what a cat is.

But but but maybe, just __maybe__ a group of MOSFETs understand in some fashion, `feature_xyz` about a cat.
Ooo. What if each group of MOSFETs find some `feature_xyz` to confidently talk about.
Ooo. A group of groups of MOSFETs does confidently tell you a coherent story about a "cat", even if each group has no idea what a cat is.

## Now actually talking about this in programming terms
So I've been circling the same concept, but if I lay it out cleanly, the best way I understand "Machine Learning" is:
> A system can learn because a system is \*inherently able to pick out things it can confidently argue about, and a group of confidence scores build a coherent picture of what `f(input) = output` is.
> \* I've used inherently very loosely here, not every system can inherently learn, and that is a good thing, because it makes teaching electrons hard.

### So how do you _actually_ implement this.. and why python?
Well I could've done this in Rust for all it matters to me, but the unglorious answer is, it's easier to debug python and write python

Cont. @ `v0.md`