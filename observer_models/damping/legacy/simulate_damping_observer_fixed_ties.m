clearvars; close all;


load_dir = "./expt_stim_exp7new_v1t_try1";

for ii =1:200
    for jj = 1:525
        RT60_vals = load(load_dir+"/"+num2str(ii)+"/"+num2str(jj)+".mat");
        stim_key = readmatrix(load_dir+"/"+num2str(ii)+"/stim_key.csv");
        
        RT60_1 = RT60_vals.RT60_1;
        RT60_2 = RT60_vals.RT60_2;
        RT60_1 = sort(RT60_1, 'descend');
        RT60_2 = sort(RT60_2, 'descend');
        if mean(RT60_1(1:10)) > mean(RT60_2(1:10))
            decision(ii,jj) = 0;
            
        else
            decision(ii,jj) = 1;
        end
    end

    mod_ans(ii,:) = 1-decision(ii,:);

    % Divide mod_ans into conditions based on stim_key(:,3)
    for kkk = 1:5
        for ijk = [1 2 3 4 5 6 7]
            ind = find(stim_key(:,3)==kkk & stim_key(:,1)==ijk);
            mod_ans_cond(ii,kkk,ijk) = mean(mod_ans(ii,ind));
            mod_ans_all(ii,kkk,ijk,:) = mod_ans(ii,ind);
        end
    end
    
        
end

save("model_result_200_v3.mat","mod_ans_cond","mod_ans_all")
