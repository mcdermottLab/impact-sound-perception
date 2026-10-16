clearvars; close all
load_dir = "../data/modes_synth";
conds =0:5;
materials = "wood";
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
            modes(3,:)=modes(3,:)*RT60_mults(kkk);
           
            if material=="metal" | material=="ceramic" |material == "stone" |material == "glass" 
                modes(3,:)=modes(3,:)*1.8;
                mat_res  = 1;
                vel = 4;
            else
                modes(3,:)=modes(3,:)*1.8;
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
           
            
            num_modes = 10;
        
            mass = 0.1;
            stiff = 8e7;
            if material == "plastic" | material == "cardboard" | material == "composite"
                stiff = 5e6;
            end 
            for ii =1:100

              
                
                if cond == "50"
                    [IR_fin,ff] = synthesize_ir_inverted_distribution(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);
                elseif cond =="5"
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
                    [IR_fin,ff] = synthesize_ir_from_gaussian_statistics(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res);
                else
                    [IR_fin,ff] = synthesize_ir_from_fitted_modes(modes,modes_noise,mat_res);
                    
                end

                y_fin = generate_impact_sound(IR_fin,44100,mass,stiff,vel,5,1);
                savedir = load_dir + "/resynth_exp7new_v1_try3/"+string(jjj+1)+"/"+num2str(damps_cond(kkk))+"/"+num2str(size_bucket)+"/" + material + "/";
                if ~exist(savedir, 'dir')
                     mkdir(savedir);  % create the directory if it doesn't exist
                end
                mass_mult = (mass/0.1)^0.75;
                stiff_mult = (0.1/mass)^0.5;

                fs=44100;
                audiowrite(savedir+string(ii)+".wav",y_fin/max(abs(y_fin)),fs)
        
           
            end


                
            end
            end
        end
    end
