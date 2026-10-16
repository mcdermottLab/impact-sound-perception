clearvars; close all
load_dir = "../data/modes_synth";
conds = 5:6;
materials = ["ceramic","glass","metal","stone","wood"];
amps = [1;1;1;1;1];
size_buckets = [1:3];
damps_cond = [0,0.25,0.5,0.75,0.9];
RT60_mults = 1.25*(1-damps_cond);


for iii = 1:length(materials)   % for materials
    material = materials(iii);
    for kkk = 1:length(damps_cond) % for damping conditions
        for jjj = conds    % for  condition numbers
            for jkl =1:length(size_buckets)
            cond = string(jjj);
            size_bucket = size_buckets(jkl);

            load(load_dir+"/"+material+"_"+size_bucket+"_modes.mat")
            
            
           
            mat_res = 0;
            if cond ~= "5"
            modes(3,:)=modes(3,:)*RT60_mults(kkk);
            end
           
            if material=="metal" | material=="ceramic" |material == "stone" |material == "glass" 
                modes(3,:)=modes(3,:)*1.8;
                mat_res  = 1;
                vel = 4;
            else
                modes(3,:)=modes(3,:)*1.1;
                vel = 4;
            end
            M_m = nanmean(modes,2);
            M_c= nancov(modes');
            n_mode_noise = size(modes_noise,2)/2;
            alpha = modes_noise(:,1:n_mode_noise);
            gamma = modes_noise(:,(n_mode_noise +1):(2*n_mode_noise));
            
            modes_noise = zeros([size(alpha,1) size(alpha,2) 2]);
            modes_noise(:,:,1) = alpha;
            modes_noise(:,:,2) = gamma;
            
            M_std = squeeze(nanstd(modes_noise,1));
            
            %Collapse the last dim of modes_noise
            modes_noise = reshape(modes_noise,[size(modes_noise,1) size(modes_noise,2)*size(modes_noise,3)]);
            M_m_noise = squeeze(nanmean(modes_noise,1));
            M_c_noise = nancov(modes_noise);
            M_c_noise = diag(diag(M_c_noise));
        
            if cond == "3"
                M_c = -M_c+2*diag(diag(M_c));
                M_m_noise(1:n_mode_noise) = fliplr(M_m_noise(1:n_mode_noise)); 
                M_m_noise((n_mode_noise+1):end) = fliplr(M_m_noise((n_mode_noise+1):end));                
                M_c_noise = 5* M_c_noise;
                M_c = 5* M_c;
                
            end
            
           if cond == "2"
                % change the sign of the offdiagonal elements of M_c
                M_c = -M_c+2*diag(diag(M_c));
                M_m_noise(1:n_mode_noise) = fliplr(M_m_noise(1:n_mode_noise)); 
                M_m_noise((n_mode_noise+1):end) = fliplr(M_m_noise((n_mode_noise+1):end));

           end
           
           if cond == "5"
                M_c = -M_c+2*diag(diag(M_c));
                M_m_noise(1:n_mode_noise) = -90-M_m_noise(1:n_mode_noise); 
    
                max_val = [16000;-60;1];
                max_val(2)= min(modes(2,:));
                M_m = max_val-M_m; 
                M_m(3) = M_m(3)*RT60_mults(kkk);
           % Reduce all the values in the third column and row of M_c by sqrt of RT60_mults
           sqrt_RT60_mults = sqrt(RT60_mults(kkk));
           M_c(3, :) = M_c(3, :) / sqrt_RT60_mults;
           M_c(:, 3) = M_c(:, 3) / sqrt_RT60_mults;
           
           end
            
            num_modes = 10;
        
            mass = 0.1;
            stiff = 8e7;
            if material == "plastic" | material == "cardboard" | material == "composite"
                stiff = 5e6;
            end 
            for ii =1:100

              
                
                if cond == "5"
                    [modes1,modes_noise1] = IR_from_stats_inv_v5t(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);
                elseif cond =="6"
                    [modes1,modes_noise1] = IR_from_stats_lin_v5t(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);
                elseif cond == "4"
                    min_mode = min(modes,[],2);
                    max_mode = max(modes,[],2);
                    modes1 = ((max_mode-min_mode).*rand([1 10])+min_mode)';
                   
                elseif cond == "3"
                    %Inverted covrariance + Scaled variance
                    [modes1,modes_noise1] = IR_from_stats_v5t(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);

                elseif cond == "2"
                    %Inverted Covariance
                    [modes1,modes_noise1] = IR_from_stats_v5t(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);
                elseif cond =="1"
                    [modes1,modes_noise1] = IR_from_stats_v5t(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);
                else
                    [modes1,modes_noise1] = IR_from_modes_v5t(modes,modes_noise);
                    
                end

                
                savedir = load_dir + "/resynth_exp7new_v1t_try2/"+string(jjj+1)+"/"+num2str(damps_cond(kkk))+"/"+num2str(size_bucket)+"/" + material + "/";
                if ~exist(savedir, 'dir')
                     mkdir(savedir);  % create the directory if it doesn't exist
                end
            if ii == 1
            all_modes = zeros(100,size(modes1,1), size(modes1,2)); % Preallocate for 100 iterations
            end
        
            all_modes(ii,:,:) = modes1;
        
           
            end

        save(savedir+"/all_modes.mat", "all_modes");

                
            end
            end
        end
    end
