function [sound_fin,ff] = synthesize_ir_from_gaussian_statistics(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res)

Fs = 44100;

num_modes_noise = length(M_m_noise)/2;
for ijk = 1:10
    modes = mvnrnd(M_m,M_c,num_modes);
    modes_noise = mvnrnd(M_m_noise,M_c_noise,1); 
    sin_pdf1 =0;
    for jkl =1:num_modes
        sin_pdf = log(mvnpdf(modes(jkl,:)',M_m,M_c));
        sin_pdf = sin_pdf+ sin_pdf1;
        sin_pdf1 = sin_pdf;
    end
    noise_pdf = log(mvnpdf(modes_noise,M_m_noise,M_c_noise));
    
    if ijk>1
        if sin_pdf1>prev_sin
            modes_f = modes;
            prev_sin = sin_pdf1;
        end

        if noise_pdf>prev_noise
            modes_f = modes;
            prev_noise = noise_pdf;
        end

    else
        prev_sin = sin_pdf1;
        prev_noise = noise_pdf;
        modes_f = modes;
        modes_noise_f = modes_noise;
    end

end






mode_freq = abs(modes(:,1));
mode_amp = modes(:,2);
mode_RT60 = (modes(:,3));
alpha = modes_noise(:,1:(num_modes_noise));
gamma = modes_noise(:,(num_modes_noise +1):(2*num_modes_noise));
num_modes = size(modes,1);
mode_amp(mode_RT60<0.00)=-120;

mode_RT60(mode_RT60<0.00)=0.002;
gamma(gamma<0.00)=0.002;
sound_f  = zeros([1 Fs]);
noise_F = zeros([num_modes 1*Fs]);
tt = 1/Fs:1/Fs:1;
tt1 = 1/Fs:1/Fs:3;
Npts = length(tt);

noise_samp = randn(size(sound_f));
noise_samp = noise_samp/max(abs(noise_samp));
[fltbnk,ff,erbff]=make_erb_cos_filters(1*Npts,Fs,num_modes_noise-2,50,16000);
noise_sub = generate_subbands(noise_samp,fltbnk);
for jj = 1:num_modes
    sound_f = sound_f + 10.^(mode_amp(jj)/20).*10.^(-(60/20)*tt/mode_RT60(jj)).*sin(2*pi*mode_freq(jj)*tt); 
end


for jj = 1:num_modes_noise
    noise_F(jj,:) = 10.^(alpha(jj)/20).*10.^(-3*tt/gamma(jj)).*noise_sub(:,jj)'./max(abs(noise_sub(jj)))*0.1;
end

if mat_res == 1
    for jj = 1:num_modes_noise
        noise_F(jj,:) = 10.^(alpha(jj)/20).*10.^(-3*tt/gamma(jj)).*noise_sub(:,jj)'./max(abs(noise_sub(jj)))*0.01;
    end
end



noise_f = sum(noise_F,1)';

sound_f = sound_f + noise_f';

sound_f =sound_f - mean(sound_f);

sound_fin = sound_f/max(abs(sound_f));

end