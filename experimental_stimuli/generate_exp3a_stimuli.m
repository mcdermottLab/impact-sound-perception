clearvars; close all
load_dir = "../data/modes_synth";
% For Exp3a: Same object on each trial (only 1 impulse response per trial)
conds = 0:6;
materials = ["wood","glass","plastic","metal","stone","ceramic"];
size_buckets = 3:4;
masses = [0.1 10^-0.75 10.^(linspace(-0.5,1,4))];


for iii = 1:length(materials)   % for materials
    material = materials(iii);
    for kkk = 1 % for size buckets
        for jjj = conds    % for condition numbers
            for jkl = 1:length(masses)   % for mass
            cond = string(jjj);
            size_bucket = string(randi(size_buckets)); % randomly choose one of the size buckets
           
            
            load(load_dir+"/"+material+"_"+size_bucket+"_modes.mat")

            
            M_c= nancov(modes');
           
            mat_res = 0;
            if material=="metal" | material=="ceramic" |material == "stone" |material == "glass" 
                mat_res  = 1;
                
            end
            vel = 4;
            M_m = nanmean(modes,2);
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
           
           end
            
            num_modes = 10;
        
            
            mass = masses(jkl);

            
            stiff = 1e7;
            
            for ii = 1:100
                % For Exp3a: Generate ONE impulse response per trial (same object)
                if cond == "5"  
                    [IR_fin,ff] = synthesize_ir_inverted_distribution(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);
                elseif cond =="6"
                    [IR_fin,ff] = synthesize_ir_linear_decay(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);
                elseif cond == "4"
                    min_mode = min(modes,[],2);
                    max_mode = max(modes,[],2);
                    [IR_fin,ff] = synthesize_ir_uniform_modes(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res,min_mode,max_mode);
                elseif cond == "3"
                    %Inverted covrariance + Scaled variance
                    [IR_fin,ff] = synthesize_ir_from_gaussian_statistics(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);

                elseif cond == "2"
                    %Inverted Covariance
                    
                    [IR_fin,ff] = synthesize_ir_from_gaussian_statistics(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);
                elseif cond =="1"
                    % Full generative model
                    [IR_fin,ff] = synthesize_ir_from_gaussian_statistics(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);
                else
                    [IR_fin,ff] = synthesize_ir_from_fitted_modes(modes,modes_noise,mat_res);
                    y_fin = generate_impact_sound(IR_fin,44100,mass,stiff,vel,5,0);
                    while isnan(max(abs(y_fin)))
                        [IR_fin,ff] = synthesize_ir_from_fitted_modes(modes,modes_noise,mat_res);
                        y_fin = generate_impact_sound(IR_fin,44100,mass,stiff,vel,5,0);
                    end
                end

                % Use the SAME impulse response for both sounds in the trial
                % This is the key difference from Exp3b
                y_fin = generate_impact_sound(IR_fin,44100,mass,stiff,vel,5,0);
                
                fs = 44100;
                
                savedir = load_dir + "/resynth_exp3a/"+string(jjj+1)+"/"+num2str(mass)+"/" + material + "/";
                if ~exist(savedir, 'dir')
                     mkdir(savedir);  % create the directory if it doesn't exist
                end
                mass_mult = (mass/10);
             

               
                audiowrite(savedir+string(ii)+".wav",mass_mult*y_fin/max(abs(y_fin)),fs)
            end
            end
        end
    end
end
