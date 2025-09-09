using CSV
using DataFrames
using Dates
using Random

# Load the CSV data
df = CSV.File("spo2rr.csv") |> DataFrame
spo2_df = filter(row -> row[:item_label] == "SpO2", df)
rr_df = filter(row -> row[:item_label] == "Resp_Rate", df)

# Preprocessing

# println("Number of missing values in each column:")
# missing_count = [sum(ismissing.(df[!, col])) for col in names(df)]
# println(missing_count)

spo2_clean = filter(row -> row.value > 0 && row.value <= 100, spo2_df)
# spo2_clean.value .= spo2_clean.value ./ 100
rr_clean = filter(row -> row.value > 0 && row.value <= 30, rr_df)
# rr_clean.value .= rr_clean.value ./ 60

comb_df = innerjoin(spo2_clean, rr_clean, on = [:subject_id, :hadm_id, :charttime], makeunique=true)

select!(comb_df, Not(:itemid))
select!(comb_df, Not(:itemid_1))
select!(comb_df, Not(:item_label))
select!(comb_df, Not(:item_label_1))
rename!(comb_df, Dict(
    :value => :spo2,
    :value_1 => :resp_rate
))
rename!(comb_df, Dict(
    :valueuom => :spo2_unit,
    :valueuom_1 => :resp_rate_unit
))

println(first(comb_df, 5))

# Create a state identifier
comb_df.state = [(s, r) for (s, r) in zip(comb_df.spo2, comb_df.resp_rate)]

println(first(comb_df, 5))

struct state
    spo2::Float64        # SpO2 level
    rr::Float64   # Respiratory rate
end

const actions = [
    "increase_fio2",      # Action 1
    "decrease_fio2",      # Action 2
    "increase_rr", # Action 3
    "decrease_rr", # Action 4
    "change_mode",        # Action 5
    "trigger_alarm",      # Action 6
    "do_nothing"          # Action 7
]

states = [state(row[:spo2], row[:resp_rate]) for row in eachrow(comb_df)]

function reward(state::state, action::String)
    
    # Optimal state ranges
    optimal_spo2 = (92.0, 100.0)
    optimal_rr = (12.0, 20.0)

    # Initial reward is 0
    reward = 0.0

    if state.spo2 < optimal_spo2[1]  # SpO2 too low
        if action == "increase_fio2"
            reward += 10.0
        elseif action == "increase_rr"
            reward += 5.0
        end
    elseif state.spo2 > optimal_spo2[2]  # SpO2 too high
        if action == "decrease_fio2"
            reward += 10.0
        elseif action == "decrease_rr"
            reward += 5.0
        end
    else  # SpO2 within target range
        if action == "do_nothing"
            reward += 1.0  # Encourage maintaining stable levels
        end
    end

    # Evaluate reward for Respiratory Rate
    if state.rr < optimal_rr[1]  # Respiratory rate too low
        if action == "increase_rr"
            reward += 10.0
        end
    elseif state.rr > optimal_rr[2]  # Respiratory rate too high
        if action == "decrease_rr"
            reward += 10.0
        end
    else  # Respiratory rate within target range
        if action == "do_nothing"
            reward += 1.0
        end
    end

    # Penalize inappropriate actions
    if (state.spo2 < optimal_spo2[1] && action == "decrease_fio2") ||
       (state.spo2 > optimal_spo2[2] && action == "increase_fio2") ||
       (state.rr < optimal_rr[1] && action == "decrease_rr") ||
       (state.rr > optimal_rr[2] && action == "increase_rr")
        reward -= 15.0  # Penalize for moving further from target range
    end

    return reward
end

function find_nearest_state(new_state::state, states::Vector{state})
    min_dist = Inf
    nearest_state = states[1]

    for s in states
        dist = sqrt((s.spo2 - new_state.spo2)^2 + (s.rr - new_state.rr)^2)
        if dist < min_dist
            min_dist = dist
            nearest_state = s
        end
    end

    return nearest_state
end

function transition(s::state, action::String, states::Vector{state})
    new_spo2 = s.spo2
    new_rr = s.rr

    if action == "increase_fio2"
        new_spo2 += 2.0
    elseif action == "decrease_fio2"
        new_spo2 -= 2.0
    elseif action == "increase_rr"
        new_rr += 1.0
    elseif action == "decrease_rr"
        new_rr -= 1.0
    elseif action == "change_mode"
        return s
    elseif action == "trigger_alarm"
        return s
    elseif action == "do_nothing"
        return s
    end

    new_spo2 = clamp(new_spo2, 0.0, 100.0)
    new_rr = clamp(new_rr, 0.0, 60.0)

    new_state = state(new_spo2, new_rr)

    return find_nearest_state(new_state, states)
end

function q_learning(q_table, episodes; α=0.1, γ=0.95, ε=0.1)
    for episode in 1:episodes
        state = states[rand(1:length(states))]
        
        for t in 1:500  
            if rand() < ε
                action = actions[rand(1:length(actions))]  # Explore
            else
                action = argmax(q_table[(state)]) # Exploit
            end

            new_state = transition(state, action, states)
            r = reward(new_state, action)

            max_future_q = maximum(values(q_table[new_state]))
            q_table[state][action] += α * (r + γ * max_future_q - q_table[state][action])

            state = new_state
        end
    end
    return q_table
end

function derive_policy(q_table)
    policy = Dict{state, String}()
    for (state, action_values) in q_table
        policy[state] = argmax(action_values)
    end
    return policy
end

# Define state discretization
# spo2_range = 0:2:100  # Discretize SpO2 levels from 0 to 100 (step of 2)
# rr_range = 0:2:60     # Discretize Respiratory Rate from 0 to 60 (step of 2)

# Create all possible (state, action) combinations
q_table = Dict{state, Dict{String, Float64}}()

for state in states
    q_table[state] = Dict(action => 0.0 for action in actions)
end

q_table = q_learning(q_table, 2000)

# println("Final Q-values:")
# for state in keys(q_table)
#     println("State: $state, Q-values: $(q_table[state])")
# end

optimal_policy = derive_policy(q_table)

function save_policy_to_csv(optimal_policy::Dict{state, String}, filename::String="optimal_policy1.csv")

    policy_data = [(s.spo2, s.rr, action) for (s, action) in optimal_policy]
    policy_df = DataFrame(
        SpO2 = [row[1] for row in policy_data],
        Respiratory_Rate = [row[2] for row in policy_data],
        Recommended_Action = [row[3] for row in policy_data]
    )
    
    sort!(policy_df, [:SpO2, :Respiratory_Rate])
    
    CSV.write(filename, policy_df)
    
    println("Policy saved to $filename")
    
    println("\nFirst 10 rows of the policy:")
    println(first(policy_df, 10))
    
    return policy_df
end

policy_dataframe = save_policy_to_csv(optimal_policy)

# println("Optimal Policy:")
# for (state, action) in optimal_policy
#     println("In state '$state', take action '$action'")
# end
