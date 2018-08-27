# using CuArrays
using Distributions: Uniform
using Flux, Flux.Tracker
import Flux.params
using OpenAIGym
import Reinforce.action

mutable struct OUNoise
    μ  # mu
    θ  # theta
    σ  # sigma
    X  # input
end

# Define custom policy for choosing action
mutable struct PendulumPolicy <: Reinforce.AbstractPolicy
    train::Bool

    function PendulumPolicy(train=true)
        new(train)
    end
end

env = GymEnv("Pendulum-v0")

n_states = length(env.state)
n_actions = length(env.actions)  # Action is continuous in case of Pendulum
action_bound = Float32(env.actions.hi[1])
batch_size = 128
mem_size = 1e6
γ = 0.99f0  # gamma, discount rate
τ = 0.01f0  # tau, for running average while updating target networks
η_actor = 0.0001f0  # eta, learning rate, actor
η_critic = 0.001f0  # eta, learning_rate, critic
max_episodes = 2000
max_episode_length = 1000
max_frames = 12000
memory = []
frames = 0
show = true

w_init(dims...) = rand(Uniform(-0.003f0, 0.003f0), dims...)

noise = OUNoise(0.0f0, 0.15f0, 0.2f0, [0.0f0])

actor = Chain(
    Dense(n_states, 256, relu),
    Dense(256, 256, relu),
    Dense(256, n_actions, tanh, initW=w_init, initb=w_init),
    x -> x * action_bound
) |> gpu
actor_target = deepcopy(actor)

critic = Chain(
    Dense(n_states + n_actions, 256, relu),
    Dense(256, 256, relu),
    Dense(256, 1, initW=w_init, initb=w_init)
) |> gpu
critic_target = deepcopy(critic)

optimizer_actor = ADAM(params(actor), η_actor)
optimizer_critic = ADAM(params(critic), η_critic)


function train()
    # Getting data in shape
    minibatch = sample(memory, batch_size)
    x = hcat(minibatch...)
    s = hcat(x[1, :]...)
    a = hcat(x[2, :]...)
    r = hcat(x[3, :]...) |> gpu
    s′ = hcat(x[4, :]...) |> gpu
    s_mask = .!hcat(x[5, :]...) |> gpu

    # Update critic
    a′ = actor_target(s′).data
    crit_tgt_in = vcat(s′, a′)
    v′ = critic_target(crit_tgt_in).data
    y = r + γ * v′ .* s_mask  # set v′ to θ where s_ is terminal state
    crit_in = vcat(s, a) |> gpu
    v = critic(crit_in)
    loss_crit = Flux.mse(y, v)

    # Update Actor
    actions = actor(s |> gpu)
    crit_in = param(vcat(s |> gpu, actions.data))
    crit_out = critic(crit_in)
    Flux.back!(sum(crit_out))
    actor_grads = -crit_in.grad[end, :]
    zero_grad!(actor)
    Flux.back!(actions, actor_grads)  # Chain rule
    optimizer_actor()

    zero_grad!(critic)
    Flux.back!(loss_crit)
    optimizer_critic()
end

""" Choose action according to policy PendulumPolicy """
function action(π::PendulumPolicy, reward, state, action)
    state = reshape(state, size(state)..., 1)
    actor_pred = actor(state |> gpu).data + action_bound * sample_noise(noise)[1] * π.train
    clamp.(actor_pred[:, 1], -action_bound, action_bound) |> gpu  # returns action
end

function episode!(env, π=RandomPolicy())
    global frames
    ep = Episode(env, π)  # Run episode with policy π
    frm = 0
    for (s, a, r, s′) in ep
        if show
            OpenAIGym.render(env)
        end
        r = env.done ? -1 : r
        if π.train
            remember(s, a, r, s′, env.done)
        end
        frames += 1
        frm += 1
        if length(memory) >= batch_size && π.train
            train()
            update_target!(actor_target, actor; τ=τ)
            update_target!(critic_target, critic; τ=τ)
        end
    end
    ep.total_reward
end

function nullify_grad!(p)
    if typeof(p) <: TrackedArray
        p.grad .= 0.0f0
    end
    p
end

""" Stores the tuple of state, action, reward, next_state, and done """
function remember(state, action, reward, next_state, done)
    if length(memory) >= mem_size
        deleteat!(memory, 1)
    end
    push!(memory, [state, action, reward, next_state, done])
end

function sample_noise(noise::OUNoise)
    dx = noise.θ * (noise.μ - noise.X)
    dx += noise.σ * randn(Float32, length(noise.X))
    noise.X += dx
end

function train()
    # Getting data in shape
    minibatch = sample(memory, batch_size)
    x = hcat(minibatch...)
    s = hcat(x[1, :]...)
    a = hcat(x[2, :]...)
    r = hcat(x[3, :]...) |> gpu
    s′ = hcat(x[4, :]...) |> gpu
    s_mask = .!hcat(x[5, :]...) |> gpu

    # Update critic
    a′ = actor_target(s′).data
    crit_tgt_in = vcat(s′, a′)
    v′ = critic_target(crit_tgt_in).data
    y = r + γ * v′ .* s_mask  # set v′ to θ where s_ is terminal state
    crit_in = vcat(s, a) |> gpu
    v = critic(crit_in)
    loss_crit = Flux.mse(y, v)

    # Update Actor
    actions = actor(s |> gpu)
    crit_in = param(vcat(s |> gpu, actions.data))
    crit_out = critic(crit_in)
    Flux.back!(sum(crit_out))
    actor_grads = -crit_in.grad[end, :]
    zero_grad!(actor)
    Flux.back!(actions, actor_grads)  # Chain rule
    optimizer_actor()

    zero_grad!(critic)
    Flux.back!(loss_crit)
    optimizer_critic()
end

function update_target!(target, model; τ=1.0f0)
    for (p_t, p_m) in zip(params(target), params(model))
        p_t.data .= (1.0f0 - τ) * p_t.data .+ τ * p_m.data
    end
end

function zero_grad!(model)
    model = mapleaves(nullify_grad!, model)
end


# ---------- Training ----------
scores = zeros(100)
episode_count = 1
idx = 1
while episode_count <= max_episodes
    reset!(env)
    total_reward = episode!(env, PendulumPolicy())
    scores[idx] = total_reward
    idx = idx % 100 + 1
    avg_reward = mean(scores)
    println("Episode: $episode_count | Score: $total_reward | Avg score: $avg_reward | Frames: $frames")
    episode_count += 1
end

# ---------- Testing  ----------
