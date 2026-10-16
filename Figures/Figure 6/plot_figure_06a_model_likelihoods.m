clearvars; close all
load_dir = "../../data/modes_synth";

conds = 1:7;
materials = ["glass","ceramic","metal","stone","wood","plastic","cardboard"];
conditions = ["Recorded","Full generative model","Inverted Covariances","Scaled variance","Uniform random","Inverted distribution","2 modes"];
amps =[0.5; 0.8; 0.2; 0.99; 0.5; 0.97; 0.3];

size_buckets = 1:4;
masses = [0.1];
n_samples = 10000;

% Initialize arrays to store log likelihoods for each condition and size bucket
log_likelihood_means = zeros(length(conds),length(size_buckets),length(materials));
log_likelihood_vars = zeros(length(conds), length(size_buckets), length(materials));

n_mode_noise =20;

% Initialize arrays to store modes and parameters for all materials
all_M_m = zeros(3, length(materials));
all_M_c = zeros(3, 3, length(materials));
all_M_m_noise = zeros(length(materials), n_mode_noise, 2);
all_M_c_noise = zeros(length(materials), n_mode_noise, 2, 2);

% First loop to load all material data
for iii = 1:length(materials)
    material = materials(iii);
    for kkk = 1:2
        size_bucket = size_buckets(kkk+2);
        load(load_dir+"/"+material+"_"+size_bucket+"_modes.mat")
        
        % Store parameters for each material
        all_M_m(:,iii) = nanmean(modes,2);
        all_M_c(:,:,iii) = nancov(modes');
        
        % Process noise modes similar to original code
        n_mode_noise = size(modes_noise,2)/2;
        alpha = modes_noise(:,1:n_mode_noise);
        gamma = modes_noise(:,(n_mode_noise+1):(2*n_mode_noise));
        
        modes_noise = zeros([size(alpha,1) size(alpha,2) 2]);
        modes_noise(:,:,1) = alpha;
        modes_noise(:,:,2) = gamma;
        
        for mode = 1:n_mode_noise
            mode_data = [squeeze(modes_noise(:,mode,1)) squeeze(modes_noise(:,mode,2))];
            all_M_m_noise(iii,mode,:) = nanmean(mode_data)';
            all_M_c_noise(iii,mode,:,:) = nancov(mode_data);
        end
    end
end

M_m_overall = nanmean(all_M_m,2);
M_c_overall = nanmean(all_M_c,3);
M_m_noise_overall = squeeze(nanmean(all_M_m_noise,1));
M_c_noise_overall = squeeze(nanmean(all_M_c_noise,1));

for iii = 1:length(materials)   % for materials
    material = materials(iii);
    for kkk = 1:4 % for size buckets
        for jjj = conds    % for  condition numbers
            cond = string(jjj);
            size_bucket = size_buckets(kkk); % randomly choose one of the size buckets
            
                load(load_dir+"/"+material+"_"+size_bucket+"_modes.mat")
            
            M_c= nancov(modes');
           
            % make the off diagonal elements zero
            
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
            % Initialize arrays to store means and covariances for each mode
            M_m_noise = zeros(2, n_mode_noise); % 2D means for each mode
            M_c_noise = zeros(2, 2, n_mode_noise); % 2x2 covariance matrices for each mode

            % Calculate mean and covariance for each mode
            for mode = 1:n_mode_noise
                % Extract alpha and gamma values for this mode
                mode_data = [squeeze(modes_noise(:,mode,1)) squeeze(modes_noise(:,mode,2))];
                
                % Calculate mean
                M_m_noise(:,mode) = nanmean(mode_data)';
                
                % Calculate covariance
                M_c_noise(:,:,mode) = nancov(mode_data);
            end

            % Bring last dimension forward for M_m_noise
            M_m_noise = permute(M_m_noise, [2 1]);
            
            % Bring last dimension forward for M_c_noise 
            M_c_noise = permute(M_c_noise, [3 1 2]);
            
           switch jjj
            case 1
                samples = modes';
                % Calculate likelihood for each material separately
                material_likelihoods = zeros(size(samples,1), length(materials));
                for mat = 1:length(materials)
                    material_likelihoods(:,mat) = mvnpdf(samples, all_M_m(:,mat)', all_M_c(:,:,mat));
                end
                total_likelihood = mean(material_likelihoods, 2);  % Average across materials
                
                % Calculate noise likelihoods for each material separately
                noise_samples = zeros(n_samples, n_mode_noise, 2);
                for m = 1:n_mode_noise
                    samples_for_mode = mvnrnd(M_m_noise(m,:), squeeze(M_c_noise(m,:,:)), n_samples);
                    noise_samples(:,m,:) = reshape(samples_for_mode, [n_samples, 1, 2]);
                end
                
                pts = reshape(noise_samples, [n_samples*n_mode_noise, 2]);
                all_noise_likelihoods = zeros(size(pts,1), length(materials));
                
                for mat = 1:length(materials)
                    for mode = 1:n_mode_noise
                        mode_pts = pts((mode-1)*n_samples + 1:mode*n_samples, :);
                        % Use the 2D noise parameters from all_M_m_noise and all_M_c_noise
                        mode_mean = squeeze(all_M_m_noise(mat,mode,:));  % This should be 2D
                        mode_cov = squeeze(all_M_c_noise(mat,mode,:,:)); % This should be 2x2
                        all_noise_likelihoods((mode-1)*n_samples + 1:mode*n_samples, mat) = mvnpdf(mode_pts, mode_mean', mode_cov);
                    end
                end
                noise_likelihoods = reshape(mean(all_noise_likelihoods,2), [n_samples, n_mode_noise]);

            case 2
                M_m_noise_n = M_m_noise;
                
                samples = mvnrnd(M_m, M_c, n_samples);
                % Calculate combined likelihood across all materials
                total_likelihood = mvnpdf(samples, M_m_overall', M_c_overall);
                
                % Handle noise samples and likelihoods same as case 1
                noise_samples = zeros(n_samples, n_mode_noise, 2);
                for m = 1:n_mode_noise
                    samples_for_mode = mvnrnd(M_m_noise_n(m,:), squeeze(M_c_noise(m,:,:)), n_samples);
                    noise_samples(:,m,:) = reshape(samples_for_mode, [n_samples, 1, 2]);
                end
                
                pts = reshape(noise_samples, [n_samples*n_mode_noise, 2]);
                all_likelihoods = zeros(n_samples*n_mode_noise, 1);
                for mode = 1:n_mode_noise
                    mode_pts = pts((mode-1)*n_samples + 1:mode*n_samples, :);
                    all_likelihoods((mode-1)*n_samples + 1:mode*n_samples) = mvnpdf(mode_pts, ...
                        squeeze(M_m_noise_overall(mode,:)), squeeze(M_c_noise_overall(mode,:,:)));
                end
                noise_likelihoods = reshape(all_likelihoods, [n_samples, n_mode_noise]);

            case 3
                M_m_noise_n = M_m_noise;
                M_c_n = -M_c+2*diag(diag(M_c)); 
                M_m_noise_n(1:n_mode_noise) = fliplr(M_m_noise(1:n_mode_noise)); 
                M_m_noise_n((n_mode_noise+1):end) = fliplr(M_m_noise((n_mode_noise+1):end));
                samples = mvnrnd(M_m, M_c_n, n_samples);
                % Calculate combined likelihood across all materials
                total_likelihood = mvnpdf(samples, M_m_overall', M_c_overall);
                
                % Handle noise samples and likelihoods same as case 1
                noise_samples = zeros(n_samples, n_mode_noise, 2);
                for m = 1:n_mode_noise
                    samples_for_mode = mvnrnd(M_m_noise_n(m,:), squeeze(M_c_noise(m,:,:)), n_samples);
                    noise_samples(:,m,:) = reshape(samples_for_mode, [n_samples, 1, 2]);
                end
                
                pts = reshape(noise_samples, [n_samples*n_mode_noise, 2]);
                all_likelihoods = zeros(n_samples*n_mode_noise, 1);
                for mode = 1:n_mode_noise
                    mode_pts = pts((mode-1)*n_samples + 1:mode*n_samples, :);
                    all_likelihoods((mode-1)*n_samples + 1:mode*n_samples) = mvnpdf(mode_pts, ...
                        squeeze(M_m_noise_overall(mode,:)), squeeze(M_c_noise_overall(mode,:,:)));
                end
                noise_likelihoods = reshape(all_likelihoods, [n_samples, n_mode_noise]);

            case 4

                M_m_noise_n = M_m_noise;
                M_c_n = -M_c+2*diag(diag(M_c)); 
                M_m_noise_n(1:n_mode_noise) = fliplr(M_m_noise(1:n_mode_noise)); 
                M_m_noise_n((n_mode_noise+1):end) = fliplr(M_m_noise((n_mode_noise+1):end));
                M_c_noise = 5*M_c_noise;
                M_c_n = 5*M_c_n;
                samples = mvnrnd(M_m, M_c_n, n_samples);
                % Calculate combined likelihood across all materials
                total_likelihood = mvnpdf(samples, M_m_overall', M_c_overall);
                
                % Handle noise samples and likelihoods same as case 1
                noise_samples = zeros(n_samples, n_mode_noise, 2);
                for m = 1:n_mode_noise
                    samples_for_mode = mvnrnd(M_m_noise_n(m,:), squeeze(M_c_noise(m,:,:)), n_samples);
                    noise_samples(:,m,:) = reshape(samples_for_mode, [n_samples, 1, 2]);
                end
                
                pts = reshape(noise_samples, [n_samples*n_mode_noise, 2]);
                all_likelihoods = zeros(n_samples*n_mode_noise, 1);
                for mode = 1:n_mode_noise
                    mode_pts = pts((mode-1)*n_samples + 1:mode*n_samples, :);
                    all_likelihoods((mode-1)*n_samples + 1:mode*n_samples) = mvnpdf(mode_pts, ...
                        squeeze(M_m_noise_overall(mode,:)), squeeze(M_c_noise_overall(mode,:,:)));
                end
                noise_likelihoods = reshape(all_likelihoods, [n_samples, n_mode_noise]);

            case 5
                
                % Get min and max of each mode dimension
                mode_mins = min(modes,[],2);
                mode_maxs = max(modes,[],2);
                
                % Generate uniform random samples between min and max for each dimension
                samples = zeros(n_samples, length(M_m));
                for dim = 1:length(M_m)
                    samples(:,dim) = mode_mins(dim) + (mode_maxs(dim)-mode_mins(dim)).*rand(n_samples,1);
                end
                % Calculate combined likelihood across all materials
                total_likelihood = mvnpdf(samples, M_m_overall', M_c_overall);
                
                % Handle noise samples and likelihoods same as case 1
                noise_samples = zeros(n_samples, n_mode_noise, 2);
                for m = 1:n_mode_noise
                    % Generate uniform random samples between min and max for each dimension
                    samples_for_mode = zeros(n_samples, 2);
                    for dim = 1:2
                        samples_for_mode(:,dim) = mode_mins(dim) + (mode_maxs(dim)-mode_mins(dim)).*rand(n_samples,1);
                    end
                    noise_samples(:,m,:) = reshape(samples_for_mode, [n_samples, 1, 2]);
                end
                
                pts = reshape(noise_samples, [n_samples*n_mode_noise, 2]);
                all_likelihoods = zeros(n_samples*n_mode_noise, 1);
                for mode = 1:n_mode_noise
                    mode_pts = pts((mode-1)*n_samples + 1:mode*n_samples, :);
                    all_likelihoods((mode-1)*n_samples + 1:mode*n_samples) = mvnpdf(mode_pts, ...
                        squeeze(M_m_noise_overall(mode,:)), squeeze(M_c_noise_overall(mode,:,:)));
                end
                noise_likelihoods = reshape(all_likelihoods, [n_samples, n_mode_noise]);

            case 6

                M_m_noise_n(1:n_mode_noise) = fliplr(M_m_noise(1:n_mode_noise)); 
                    M_m_noise_n((n_mode_noise+1):end) = fliplr(M_m_noise((n_mode_noise+1):end));
                    max_val = [13000;-30;0.5];
                    M_m_n = max_val-M_m;
                samples = mvnrnd(M_m_n, M_c, n_samples);
                % Calculate combined likelihood across all materials
                total_likelihood = mvnpdf(samples, M_m_overall', M_c_overall);
                
                % Handle noise samples and likelihoods same as case 1
                noise_samples = zeros(n_samples, n_mode_noise, 2);
                for m = 1:n_mode_noise
                    samples_for_mode = mvnrnd(M_m_noise_n(m,:), squeeze(M_c_noise(m,:,:)), n_samples);
                    noise_samples(:,m,:) = reshape(samples_for_mode, [n_samples, 1, 2]);
                end
                
                pts = reshape(noise_samples, [n_samples*n_mode_noise, 2]);
                all_likelihoods = zeros(n_samples*n_mode_noise, 1);
                for mode = 1:n_mode_noise
                    mode_pts = pts((mode-1)*n_samples + 1:mode*n_samples, :);
                    all_likelihoods((mode-1)*n_samples + 1:mode*n_samples) = mvnpdf(mode_pts, ...
                        squeeze(M_m_noise_overall(mode,:)), squeeze(M_c_noise_overall(mode,:,:)));
                end
                noise_likelihoods = reshape(all_likelihoods, [n_samples, n_mode_noise]);

            case 7
                samples = mvnrnd(M_m, M_c, n_samples);
                % From 3rd row onwards, change the second column entry to zero
                samples(3:end, 2) =-60;

                % Calculate combined likelihood across all materials
                total_likelihood = mvnpdf(samples, M_m_overall', M_c_overall);
                
                % Handle noise samples and likelihoods same as case 1
                noise_samples = zeros(n_samples, n_mode_noise, 2);
                for m = 1:n_mode_noise
                    samples_for_mode = mvnrnd(M_m_noise(m,:), squeeze(M_c_noise(m,:,:)), n_samples);
                    noise_samples(:,m,:) = reshape(samples_for_mode, [n_samples, 1, 2]);
                end
                
                pts = reshape(noise_samples, [n_samples*n_mode_noise, 2]);
                all_likelihoods = zeros(n_samples*n_mode_noise, 1);
                for mode = 1:n_mode_noise
                    mode_pts = pts((mode-1)*n_samples + 1:mode*n_samples, :);
                    all_likelihoods((mode-1)*n_samples + 1:mode*n_samples) = mvnpdf(mode_pts, ...
                        squeeze(M_m_noise_overall(mode,:)), squeeze(M_c_noise_overall(mode,:,:)));
                end
                noise_likelihoods = reshape(all_likelihoods, [n_samples, n_mode_noise]);

            end

            % Calculate final log likelihood using noise
            log_likelihood_n = max(sum(log(noise_likelihoods),2)/20, -100);
            log_likelihood = max(log(total_likelihood), -100);
            
            % Store the mean and variance
            log_likelihood_means(jjj,kkk, iii) = mean(log_likelihood,'omitmissing')+mean(log_likelihood_n,'omitmissing')/2;
            log_likelihood_vars(jjj,kkk, iii) = std(log_likelihood,"omitmissing")+std(log_likelihood_n,"omitmissing")/2;

        end
    end
end

% Create separate figures for each size bucket
colors = jet(length(materials)); % Get different colors for each material
for sb = 1:4
    
    % Plot individual error bars for each material (excluding condition 7)
    for m = 1:length(materials)
        vars = squeeze(log_likelihood_means(1:6,sb,m));
        std_vars = squeeze(log_likelihood_vars(1:6,sb,m));
        
    end
    
end

% Create a figure for averaged log likelihoods
figure;
set(gcf,'Units','inches','Position', [1 1 5 4]);
hold on

% Calculate means and standard errors (excluding condition 7)
overall_means = squeeze(mean(mean(log_likelihood_means(1:6,:,:), 3), 2));
overall_std = squeeze(mean(mean(log_likelihood_vars(1:6,:,:), 3), 2));

% Plot the averaged data
errorbar(conds(1:6), overall_means, overall_std, 'o-', 'LineWidth', 2, 'Color', 'k', ...
    'MarkerFaceColor', 'k', 'CapSize', 0);

title('Average Log Likelihood Across Materials and Sizes');
xlabel('Condition');
ylabel('Log Likelihood');
set(gca, "FontSize", 12)
ylim([-120 0])
grid on
xticks(conds(1:6))
xticklabels(conditions(1:6))
xtickangle(45)

% Save the figure
saveas(gcf, "../all_figs/Figure_6a.svg")

