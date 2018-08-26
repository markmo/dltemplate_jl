using Base.Iterators: partition
using Flux
using Flux: argmax, batchseq, chunk, crossentropy, onehot, throttle
using StatsBase: wsample

cd(@__DIR__)

isfile("input.txt") || download("http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt", "input.txt")

text = collect(readstring("input.txt"))
alphabet = [unique(text)..., '_']
text = map(ch -> onehot(ch, alphabet), text)
stop = onehot('_', alphabet)

N = length(alphabet)
seqlen = 50
nbatch = 50

X = collect(partition(batchseq(chunk(text, nbatch), stop), seqlen))
Y = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))

model = Chain(
    LSTM(N, 128),
    LSTM(128, 128),
    Dense(128, N),
    softmax
)

function loss(xs, ys)
    l = sum(crossentropy.(model.(xs), ys))
    Flux.truncate!(model)
    l
end

optimizer = ADAM(params(model), 0.01)
tx, ty = X[5], Y[5]
evalcb = () -> @show loss(tx, ty)

Flux.train!(loss, zip(X, Y), optimizer, cb=throttle(evalcb, 30))

function sample(model, alphabet, len; temp=1)
    Flux.reset!(model)
    buf = IOBuffer()
    ch = rand(alphabet)
    for i in 1:len
        write(buf, ch)
        ch = wsample(alphabet, model(onehot(ch, alphabet)).data)
    end
    String(take!(buf))
end

sample(model, alphabet, 1000) |> println
