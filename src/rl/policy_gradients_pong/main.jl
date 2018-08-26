#=
main:
- Julia version: 0.6.4
- Author: markmo
- Date: 2018-08-23
=#

using HDF5
using Gym

# hyperparams
n_hidden = 200        # number of hidden layer neurons
batch_size = 10       # number episodes for param update
learning_rate = 1e-3
gamma = 0.99          # discount factor for reward
decay_rate = 0.99     # decay factor for RMSProp, leaky sum of grade^2
resume = true         # flag to resume from previous checkpoint
show = true
model_path = "model.hd5"

# actions
UP = 2
DOWN = 3

env = GymEnv("Pong-v0")

function train(input_dim, model, grad_buffer, rmsprop_cache)
    # an image frame (a 210x160x3 byte array (integers from 0 to 255 giving pixel values)), 100,800 numbers total
    observation = reset!(env)
    prev_x = nothing  # used in computing the difference frame
    xs, hs, dlogps, rewards = [], [], [], []
    running_reward = nothing
    reward_sum = 0
    episode_count = 0
    while true
        if show
            render(env)
        end

        # preprocess the observation, convert 210x160x3 byte array to a 80x80 float vector
        cur_x = preprocess(observation)

        # set input to difference frame (i.e. subtraction of current and last frame) to detect motion
        x = prev_x == nothing ? zeros(input_dim) : cur_x - prev_x
        prev_x = cur_x

        # forward the policy network and sample an action from the returned probability
        up_prob, h = policy_forward(model, x)  # probability of going UP
        action = choose_action(up_prob)  # roll the dice!

        # record various intermediates (needed for backprop)
        push!(xs, x)  # observations
        push!(hs, h)  # hidden states
        y = action == UP ? 1 : 0  # a "fake label"

        # grad that encourages the action that was taken to be taken
        # (see http://cs231n.github.io/neural-networks-2/#losses)
        push!(dlogps, y - up_prob)  # grads

        # step the environment and get new observation
        # a +1 reward if the ball went past the opponent, a -1 reward if we missed the ball, or 0 otherwise
        observation, reward, done, info = step!(env, action)
        reward_sum += reward

        # record reward (must be done after call to `step` to get reward for previous action)
        push!(rewards, reward)

        if done  # an episode finished
            episode_count += 1

            # stack together all inputs, hidden states, action gradients,
            # and rewards for this episode
            epx = vcat(xs'...)          # episode x
            eph = vcat(hs'...)          # episode h
            epdlogp = vcat(dlogps'...)  # episode gradient
            epdlogp = reshape(epdlogp, length(epdlogp), 1)
            epr = vcat(rewards'...)     # episode reward
            epr = reshape(epr, length(epr), 1)
            xs, hs, dlogps, rewards = [], [], [], []  # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr, gamma)  # discounted episode reward

            # standardize the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr = normalize(discounted_epr)

            epdlogp = epdlogp .* discounted_epr  # modulate the gradient with advantage (PG magic happens here!)
            grad = policy_backward(model, eph, epx, epdlogp)
            for k in keys(model)
                grad_buffer[k] += grad[k]  # accumulate grad over batch
            end

            # perform RMSprop parameter update every batch_size episodes
            if episode_count % batch_size == 0
                update_parameters(model, grad_buffer, rmsprop_cache)
            end

            # boring book keeping
            running_reward = running_reward == nothing ? reward_sum : running_reward * 0.99 + reward_sum * 0.01
            log_episode(reward_sum, running_reward)
            if episode_count % 100 == 0
                save_model(model, model_path)
            end

            reward_sum = 0
            observation = reset!(env)
            prev_x = nothing
        end
    end
end

choose_action(up_prob) = rand() < up_prob ? UP : DOWN

create_model(input_dim) = Dict(
    "W1" => randn(n_hidden, input_dim) / sqrt(input_dim),
    "W2" => randn(n_hidden) / sqrt(n_hidden)
)

sigmoid(x) = 1. / (1. + exp(-x))

function policy_forward(model, x)
    h = model["W1"] * x      # compute hidden layer neuron activations
    h[h .< 0] = 0            # ReLU non-linearity
    logp = model["W2"]' * h  # compute log probability of going UP
    p = sigmoid(logp)        # sigmoid "squashing" function to interval [0, 1]
    p, h                     # return probability of going UP, and hidden state
end

function policy_backward(model, eph, epx, epdlogp)
    dW2 = vec(eph' * epdlogp)
    dh = epdlogp .* model["W2"]'
    dh[eph .<= 0] = 0        # backprop prelu
    dW1 = dh' * epx
    Dict("W1" => dW1, "W2" => dW2)
end

update_parameters(model, grad_buffer, rmsprop_cache) =
    for (k, v) in model
        grad = grad_buffer[k]  # gradient
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * grad.^2
        model[k] += learning_rate * grad ./ (sqrt.(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = zeros(size(v))  # reset batch gradient buffer
    end

function init_model()
    input_dim = 80 * 80
    if resume && isfile(model_path)
        model = load_model(model_path)
    else
        model = create_model(input_dim)
    end
    grad_buffer = Dict(k => zeros(size(v)) for (k, v) in model)
    rmsprop_cache = Dict(k => zeros(size(v)) for (k, v) in model)
    input_dim, model, grad_buffer, rmsprop_cache
end

load_model(model_path) = h5open(model_path, "r") do file
    read(file, "model")
end

save_model(model, model_path) = h5open(model_path, "w") do file
    write(file, "model", model)
end

""" Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
function preprocess(x)
    x = x[36:195, :, :]         # crop
    x = x[1:2:end, 1:2:end, 1]  # downsample by factor of 2
    x[x .== 144] = 0            # erase background (background type 1)
    x[x .== 109] = 0            # erase background (background type 2)
    x[x .!= 0] = 1              # everything else (paddles, ball) just set to 1
    convert(Array{Float64, 1}, vec(x))
end

function normalize(x)
    x -= mean(x)
    x /= std(x)
    x
end

""" Take 1D float array of rewards and compute discounted rewards """
function discount_rewards(rewards, gamma)
    discounted_rewards = zeros(size(rewards))
    running_add = 0
    for t in length(rewards):-1:1
        if rewards[t] != 0
            running_add = 0  # reset the sum since this was a game boundary (Pong specific!)
        end
        running_add *= gamma + rewards[t]
        discounted_rewards[t] = running_add
    end
    discounted_rewards
end

log_episode(reward_sum, running_reward) =
    @printf "Resetting env episode reward total was %f. running mean: %f\n" reward_sum running_reward


train(init_model()...)
