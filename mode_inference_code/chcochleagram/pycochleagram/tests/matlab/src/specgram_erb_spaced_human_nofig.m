%generates spectrogram with ERB-spaced frequency axis for sound <s>
%
%this version makes filters inside the function
%
%if zp is 1, signal is zero-padded
function img = specgram_erb_spaced_human(s,sr,low_lim,hi_lim,zp)

if zp==1
    if size(s,1)==1%row vector
        s = [s zeros(size(s))];
    elseif size(s,2)==1%column vector
        s = [s; zeros(size(s))];
    end
end

signal_length=length(s);

% find the number of filters to make - this will be specific to human
N = floor(freq2erb(hi_lim) - freq2erb(low_lim)) - 1;

[filts,Hz_cutoffs,freqs] = make_erb_cos_filts_double2(length(s), sr, N, low_lim, hi_lim); % make audiobank filter

filts = filts(:,3:end-2); % using 2x oversampled filter, so throw out 2 lowpass and 2 highpass filters

subbands = generate_subbands(s,filts);
if zp==1
    subbands = subbands(1:end/2,:);
    signal_length = signal_length/2;
end

thresh = -60;
env_sr = 6000;


sub_envs = abs(hilbert(subbands));

sub_envs = resample(sub_envs,env_sr,sr);

sub_envs(find(sub_envs<0)) = eps;

sub_envs = 20*log10(sub_envs/max(max(sub_envs)));


blur_level = 1;


img = flipud(sub_envs');


if N<=40
    tick_spacing = 5;
else
    tick_spacing = 10;
end

if length(sub_envs)/env_sr < 2
    time_spacing = .4;
else
    time_spacing = 1;
end


