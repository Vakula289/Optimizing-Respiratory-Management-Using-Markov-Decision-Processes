using CSV
using DataFrames
using Plots

# Load the optimal policy from the saved file
policy_dataframe = CSV.File("optimal_policy1.csv") |> DataFrame

using Plots
using DataFrames
using CSV

function visualize_policy(policy_dataframe::DataFrame)
    # Action color mapping
    action_colors = Dict(
        "increase_fio2" => :red,
        "decrease_fio2" => :blue,
        "increase_rr" => :green,
        "decrease_rr" => :purple,
        "change_mode" => :orange,
        "trigger_alarm" => :yellow,
        "do_nothing" => :gray
    )
    
    # Create scatter plot
    p = scatter(
        policy_dataframe.SpO2, 
        policy_dataframe.Respiratory_Rate, 
        group=policy_dataframe.Recommended_Action,
        title="State-Action Visualization",
        xlabel="SpO2 Level",
        ylabel="Respiratory Rate",
        legend=:best,
        markercolor = [action_colors[action] for action in policy_dataframe.Recommended_Action],
        marker = :circle,
        markersize = 5
    )
    
    display(p)
    savefig(p, "state_action_scatter1.png")
    return p
end

function visualize_policy_heatmap(policy_dataframe::DataFrame)
    # Create unique mappings
    unique_actions = unique(policy_dataframe.Recommended_Action)
    action_to_numeric = Dict(action => i for (i, action) in enumerate(unique_actions))
    
    # Prepare data for heatmap
    spo2_levels = policy_dataframe.SpO2
    rr_levels = policy_dataframe.Respiratory_Rate
    action_numeric = [action_to_numeric[action] for action in policy_dataframe.Recommended_Action]
    
    # Create heatmap
    p = scatter(
        spo2_levels, 
        rr_levels, 
        zcolor=action_numeric,
        title="Optimal Actions Heatmap",
        xlabel="SpO2 Level",
        ylabel="Respiratory Rate",
        color=:viridis,
        marker_z=action_numeric,
        colorbar_title="Actions",
        colorbar_labels=unique_actions,
        marker = :square,
        markersize = 8
    )
    
    display(p)
    savefig(p, "policy_heatmap1.png")
    return p
end

function action_distribution(policy_dataframe::DataFrame)
    # Count of actions
    action_counts = combine(groupby(policy_dataframe, :Recommended_Action), nrow => :Count)
    
    # Pie chart of action distribution
    p = pie(
        action_counts.Recommended_Action, 
        action_counts.Count,
        title="Distribution of Recommended Actions",
        labels=action_counts.Recommended_Action
    )
    
    display(p)
    savefig(p, "action_distribution_pie1.png")
    return p
end

function policy_surface_plot(policy_dataframe::DataFrame)
    # Create unique mappings
    unique_actions = unique(policy_dataframe.Recommended_Action)
    action_to_numeric = Dict(action => i for (i, action) in enumerate(unique_actions))
    
    # Prepare data
    action_numeric = [action_to_numeric[action] for action in policy_dataframe.Recommended_Action]
    
    # Surface plot
    p = surface(
        policy_dataframe.SpO2, 
        policy_dataframe.Respiratory_Rate, 
        action_numeric,
        title="Policy Surface Visualization",
        xlabel="SpO2 Level",
        ylabel="Respiratory Rate",
        zlabel="Action Encoding",
        color=:viridis
    )
    
    display(p)
    savefig(p, "policy_surface_plot1.png")
    return p
end

# Usage
plot1 = visualize_policy(policy_dataframe)
plot2 = visualize_policy_heatmap(policy_dataframe)
plot3 = action_distribution(policy_dataframe)
plot4 = policy_surface_plot(policy_dataframe)