function [sound_fin,ff] = synthesize_ir_uniform_modes(M_m,M_c,M_m_noise,M_c_noise,num_modes,mat_res,varargin)

if length(varargin)>0
   min_mode =varargin{1};
   max_mode =varargin{2};
end
Fs = 44100;

num_modes_noise = length(M_m_noise)/2;

    mode_freq = max_mode(1)*rand([1 num_modes]);
    mode_amp = max_mode(2)*rand([1 num_modes]);
    mode_RT60 = max_mode(3)*rand([1 num_modes]);
    alpha = min(M_m_noise(1:20)).*rand([1 num_modes_noise]);
    gamma = max(M_m_noise(21:40))*rand([1 num_modes_noise]);
   







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

sound_f =sound_f - mean(sound_f);

sound_fin = 1*sound_f/max(abs(sound_f));

end