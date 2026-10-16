clearvars; close all;


load_dir = "./expt_stim_exp7new_v1t_try1";

% Define parameters at the start
n_iterations = 200;
n_trials = 525;
n_conditions = 5;
n_levels = 7;
trials_per_condition = 15;

% Preallocate arrays
decision = zeros(n_iterations, n_trials);
mod_ans = zeros(n_iterations, n_trials);
mod_ans_all = zeros(n_iterations, n_conditions, n_levels, trials_per_condition);

for ii = 1:n_iterations
    for jj = 1:n_trials
        RT60_vals = load(load_dir+"/"+num2str(ii)+"/"+num2str(jj)+".mat");
        stim_key = readmatrix(load_dir+"/"+num2str(ii)+"/stim_key.csv");
        
        RT60_1 = RT60_vals.RT60_1;
        RT60_2 = RT60_vals.RT60_2;
        RT60_1 = sort(RT60_1, 'descend');
        RT60_2 = sort(RT60_2, 'descend');
        if mean(RT60_1(1:10)) > mean(RT60_2(1:10))
            decision(ii,jj) = 0;
            
        elseif mean(RT60_1(1:10)) < mean(RT60_2(1:10))
            decision(ii,jj) = 1;
        else
            decision(ii,jj) = randi(2)-1;
        end
    end

    mod_ans(ii,:) = 1-decision(ii,:);

    % Divide mod_ans into conditions based on stim_key(:,3)
    for kkk = 1:5
        for ijk = [1 2 3 4 5 6 7]
            ind = find(stim_key(:,3)==kkk & stim_key(:,1)==ijk);
            mod_ans_all(ii,kkk,ijk,:) = mod_ans(ii,ind);
        end
    end
    
        
end

save("model_result_200_v2.mat","mod_ans_all")
