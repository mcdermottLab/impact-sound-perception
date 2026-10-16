clearvars; close all
load_dir = "../data/modes_synth";
conds = 0:7;
materials = ["wood","glass","plastic","metal","stone","ceramic"];
amps =[0.5; 0.8; 0.2; 0.99; 0.5; 0.97; 0.3];
size_buckets = 1:4;
masses = [0.1 10^-0.75 10.^(linspace(-0.5,1,4))];
stiffness_vals = readmatrix("./stiffness_values.csv")*1e7;


for iii = 1:length(materials)   % for materials
    material = materials(iii);
    for kkk = size_buckets % for size buckets
        for jjj = conds
            for jkl =1:length(masses)   % for mass
            cond = string(jjj);
            size_bucket = string(kkk);
           
            
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
        
            if cond == "4"
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
           if cond == "6"
                M_c = -M_c+2*diag(diag(M_c));
                M_m_noise(1:n_mode_noise) = -90-M_m_noise(1:n_mode_noise); 
    
                max_val = [16000;-60;1];
                max_val(2)= min(modes(2,:));
                M_m = max_val-M_m;   
           
           end
            
            num_modes = 10;
        
            
            mass = 0.1;

            stiff = stiffness_vals(iii,kkk);
            for ii =1:100
              
                
                if cond == "6"
                    
                    [IR_fin,ff] = synthesize_ir_inverted_distribution(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);

                    

                elseif cond =="7"
                    [IR_fin,ff] = synthesize_ir_linear_decay(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);
                elseif cond == "5"
                    min_mode = min(modes,[],2);
                    max_mode = max(modes,[],2);
                    [IR_fin,ff] = synthesize_ir_uniform_modes(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res,min_mode,max_mode);
                elseif cond == "4"
                    %Inverted covrariance + Scaled variance
                    [IR_fin,ff] = synthesize_ir_from_gaussian_statistics(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);

                elseif cond =="3"
                    [IR_fin,ff] = IR_from_stats_v4n(M_m,M_c,M_m_noise,M_c_noise,2,mat_res);
                elseif cond == "2"
                    %Inverted Covariance
                    [IR_fin,ff] = synthesize_ir_from_gaussian_statistics(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);
                elseif cond =="1"
                    [IR_fin,ff] = synthesize_ir_from_gaussian_statistics(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);
                else
                    [IR_fin,ff] = synthesize_ir_from_fitted_modes(modes,modes_noise,mat_res);
                    y_fin = generate_impact_sound(IR_fin,44100,mass,stiff,vel,5,0);
                    while isnan(max(abs(y_fin)))
                        [IR_fin,ff] = synthesize_ir_from_fitted_modes(modes,modes_noise,mat_res);
                        y_fin = generate_impact_sound(IR_fin,44100,mass,stiff,vel,5,0);
                    end
                end

            
                y_fin = generate_impact_sound(IR_fin,44100,mass,stiff,vel,5,0);
               
                
                fs =44100;
                
                savedir = load_dir + "/resynth_exp1new_v1/"+material+"/"+string(size_bucket)+"_"+cond+"/";
                if ~exist(savedir, 'dir')
                     mkdir(savedir);  % create the directory if it doesn't exist
                end
                

                audiowrite(savedir+num2str(ii)+".wav",0.8*y_fin/max(abs(y_fin)),fs)
            end
            end
        end
    end
end