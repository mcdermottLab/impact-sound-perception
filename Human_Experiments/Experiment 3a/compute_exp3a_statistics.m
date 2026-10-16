function result = compute_exp3a_statistics(nResamples)
% COMPUTE_EXP3A_STATISTICS Recorded comparisons with participant-bootstrap ratio CIs.
if nargin < 1, nResamples = 10000; end
validateattributes(nResamples, {'numeric'}, {'scalar','integer','positive'});
folder = fileparts(mfilename('fullpath'));
sourceFile = fullfile(folder,'Exp3a_data.mat');
data = load(sourceFile,'resp_cond');
responses = double(data.resp_cond);
assert(size(responses,2)==7 && size(responses,3)==5);
assert(all(ismember(responses(:),[0 1])), 'Expected complete binary responses.');
names = ["Recorded","Full generative model","Inverted covariances", ...
    "Scaled variances","Uniform model","Inverted distribution","Linear decay"];
participantData = mean(responses,4);
result = threshold_inference(participantData,names,[1.8 3.16 10 31.6 100], ...
    0.8,nResamples,260933,"3a",0);
result.sourceFile = sourceFile;
metadata = readtable(fullfile(folder,'..','..','participant_data','Exp3a_metadata.csv'));
result.demographics = struct('NMetadata',height(metadata),'MeanAge',mean(metadata.Age,'omitnan'), ...
    'SDAge',std(metadata.Age,'omitnan'));
result.table.AnalysisSet = repmat("Additional comparison",height(result.table),1);
result.table.TestType = repmat("Participant permutation test",height(result.table),1);
result.table.StatisticName = repmat("Threshold difference",height(result.table),1);
disp(result.table);
end

function result = threshold_inference(participantData,names,levels,target,nResamples,seed,experiment,rateTrials)
% ParticipantData is N x condition x level (accuracy) or N x condition x
% [false-alarm rate, hit rates]. The same estimator is used for every draw.
n = size(participantData,1); nc = size(participantData,2);
stream = RandStream('mt19937ar','Seed',seed);
curves = make_curves(participantData,rateTrials);
[thresholds,lower,upper] = crossing(curves,levels,target);
observed = thresholds(2:end)-thresholds(1);
ratio = thresholds(2:end)/thresholds(1);
bootstrapThresholds = nan(nResamples,nc);
bootstrapRatioLower = zeros(nResamples,nc-1);
bootstrapRatioUpper = zeros(nResamples,nc-1);
nullDifferences = nan(nResamples,nc-1);
nullDifferenceLower = nan(nResamples,nc-1);
nullDifferenceUpper = nan(nResamples,nc-1);
for b = 1:nResamples
    indices = randi(stream,n,[n 1]);
    bootCurves = make_curves(participantData(indices,:,:),rateTrials);
    [bt,bl,bu] = crossing(bootCurves,levels,target);
    bootstrapThresholds(b,:) = bt;
    % Retain censoring bounds instead of discarding out-of-range bootstraps.
    bootstrapRatioLower(b,:) = bl(2:end)/bu(1);
    bootstrapRatioUpper(b,:) = bu(2:end)/bl(1);
    for j = 1:nc-1
        pair = participantData(:,[1 j+1],:);
        swap = rand(stream,n,1)<0.5;
        pair(swap,:,:) = pair(swap,[2 1],:);
        [permThresholds,pl,pu] = crossing(make_curves(pair,rateTrials),levels,target);
        nullDifferences(b,j) = permThresholds(2)-permThresholds(1);
        nullDifferenceLower(b,j) = pl(2)-pu(1);
        nullDifferenceUpper(b,j) = pu(2)-pl(1);
    end
end
ciLower = bound_percentile(bootstrapRatioLower,2.5);
ciUpper = bound_percentile(bootstrapRatioUpper,97.5);
invalidPermutations = sum(~isfinite(nullDifferences),1);
invalidBootstrapPairs = sum(~isfinite(bootstrapThresholds(:,2:end)) | ...
    ~isfinite(bootstrapThresholds(:,1)),1);
% Classify tail membership using interval bounds even when a threshold is
% censored. A point p-value is valid if every draw can be classified.
minimumAbsolute = max(0,max(nullDifferenceLower,-nullDifferenceUpper));
maximumAbsolute = max(abs(nullDifferenceLower),abs(nullDifferenceUpper));
definiteExtreme = minimumAbsolute >= abs(observed)-1e-12;
definiteNonExtreme = maximumAbsolute < abs(observed)-1e-12;
unresolvedPermutations = sum(~definiteExtreme & ~definiteNonExtreme,1);
extreme = sum(definiteExtreme,1);
pLower = (1+extreme)/(nResamples+1);
pUpper = (1+extreme+unresolvedPermutations)/(nResamples+1);
p = pLower;
p(unresolvedPermutations>0 | ~isfinite(observed)) = NaN;
pLower(~isfinite(observed)) = NaN; pUpper(~isfinite(observed)) = NaN;
status = repmat("ok",nc-1,1);
status(invalidBootstrapPairs>0) = "CI envelope includes censored bootstrap thresholds";
resolved = invalidPermutations>0 & unresolvedPermutations==0;
status(resolved) = status(resolved)+"; permutation tails resolved from bounds";
status(unresolvedPermutations>0) = status(unresolvedPermutations>0)+"; p bounded: ambiguous permutation tails";
status(~isfinite(ratio)) = "Observed threshold not bracketed; ratio unavailable";
Experiment = repmat(string(experiment),nc-1,1); Condition = names(2:end)';
Reference = repmat("Recorded",nc-1,1); N = repmat(n,nc-1,1);
EffectMetric = repmat("Threshold ratio: Condition / Recorded",nc-1,1);
EffectEstimate = ratio'; CILower = ciLower'; CIUpper = ciUpper';
PValue = p'; TestStatistic = observed'; DF = nan(nc-1,1); Status = status;
result.table = table(Experiment,Condition,Reference,N,EffectMetric,EffectEstimate, ...
    CILower,CIUpper,PValue,TestStatistic,DF,Status);
result.thresholds = table(names',thresholds',lower',upper', ...
    'VariableNames',{'Condition','Threshold','ThresholdLowerBound','ThresholdUpperBound'});
result.resamplingDiagnostics = table(Condition,invalidBootstrapPairs',invalidPermutations',unresolvedPermutations',pLower',pUpper', ...
    'VariableNames',{'Condition','CensoredBootstrapPairs','CensoredPermutations','AmbiguousPermutationTails','PValueLowerBound','PValueUpperBound'});
result.bootstrapThresholds = bootstrapThresholds;
result.bootstrapRatioLower = bootstrapRatioLower;
result.bootstrapRatioUpper = bootstrapRatioUpper;
result.nullDifferences = nullDifferences;
result.nullDifferenceLower = nullDifferenceLower;
result.nullDifferenceUpper = nullDifferenceUpper;
result.participantData = participantData;
result.observedCurves = curves;
result.levels = levels;
result.target = target;
result.seed = seed;
result.nResamples = nResamples;
result.method = "Condition/Recorded threshold ratio; first upward crossing with linear interpolation on the physical stimulus axis. Paired N-participant bootstrap; 95% percentile CI. For censored draws, report the envelope of percentile bounds without dropping draws. Two-sided whole-condition participant permutation on threshold differences, (b+1)/(B+1); unadjusted p-values. No extrapolation, clipping, jitter or offsets.";
end

function curves = make_curves(participantData,rateTrials)
curves = reshape(mean(participantData,1),size(participantData,2),size(participantData,3));
if rateTrials > 0
    % Correct only empirical rates of exactly 0 or 1, using half a trial.
    % Fixed N and trials per cell ensure the rule is identical in all draws.
    epsilon = 1/(2*size(participantData,1)*rateTrials);
    curves = min(max(curves,epsilon),1-epsilon);
    curves = norminv(curves(:,2:end))-norminv(curves(:,1));
end
end

function [thresholds,lower,upper] = crossing(curves,levels,target)
% No measured crossing: preserve an interval, never substitute an endpoint.
thresholds = nan(1,size(curves,1));
lower = zeros(size(thresholds)); upper = inf(size(thresholds));
for c = 1:size(curves,1)
    y = curves(c,:);
    assert(all(isfinite(y)), 'Nonfinite performance curve.');
    index = find(y>=target,1,'first');
    if isempty(index)
        lower(c) = levels(end);
    elseif y(index)==target
        thresholds(c) = levels(index);
        lower(c) = thresholds(c); upper(c) = thresholds(c);
    elseif index==1
        upper(c) = levels(1);
    else
        thresholds(c) = levels(index-1)+(target-y(index-1))* ...
            (levels(index)-levels(index-1))/(y(index)-y(index-1));
        lower(c) = thresholds(c); upper(c) = thresholds(c);
    end
end
end

function q = bound_percentile(values,percent)
% MATLAB percentile rank convention, with explicit handling of +Inf bounds.
values = sort(values,1);
n = size(values,1); rank = min(n,max(1,n*percent/100+0.5));
lo = floor(rank); hi = ceil(rank); fraction = rank-lo;
a = values(lo,:); b = values(hi,:);
q = a;
change = a~=b;
q(change) = a(change)+fraction*(b(change)-a(change));
q(isinf(b) & fraction>0) = Inf;
end
