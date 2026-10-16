function result = compute_exp3c_statistics(nPermutations)
% COMPUTE_EXP3C_STATISTICS Mean-performance test, without estimating thresholds.
% Average trials within mass ratio, then the five ratios and participants.
% Each permutation swaps an entire pair of conditions for each participant.
if nargin < 1, nPermutations = 10000; end
validateattributes(nPermutations, {'numeric'}, {'scalar','integer','positive'});
folder = fileparts(mfilename('fullpath'));
sourceFile = fullfile(folder,'Exp3c_data.mat');
data = load(sourceFile,'resp_cond');
responses = double(data.resp_cond);
assert(size(responses,2)==7 && size(responses,3)==5);
assert(all(ismember(responses(:),[0 1])), 'Expected complete binary responses.');
metadata = readtable(fullfile(folder,'..','..','participant_data','Exp3c_metadata_with_headphones.csv'));
responses = responses(1:height(metadata),:,:,:);
names = ["Recorded","Full generative model","Inverted covariances", ...
    "Scaled variances","Uniform model","Inverted distribution","Linear decay"];
participantPerformance = reshape(mean(mean(responses,4),3),size(responses,1),7);
n = size(responses,1);
differences = participantPerformance(:,2:7)-participantPerformance(:,1);
observed = mean(differences,1);
stream = RandStream('mt19937ar','Seed',260935);
nullDifferences = zeros(nPermutations,6);
bootstrapDifferences = zeros(nPermutations,6);
for j = 1:6
    % A sign flip is exactly the within-participant whole-condition exchange.
    signs = 2*(rand(stream,nPermutations,n) < 0.5)-1;
    nullDifferences(:,j) = signs*differences(:,j)/n;
end
for b = 1:nPermutations
    indices = randi(stream,n,[n 1]);
    bootstrapDifferences(b,:) = mean(differences(indices,:),1);
end
p = (1+sum(abs(nullDifferences)>=abs(observed)-1e-12,1))/(nPermutations+1);
ciLower = prctile(bootstrapDifferences,2.5,1);
ciUpper = prctile(bootstrapDifferences,97.5,1);
Experiment = repmat("3c",6,1); Condition = names(2:7)';
Reference = repmat("Recorded",6,1); N = repmat(n,6,1);
EffectMetric = repmat("Mean performance: Condition - Recorded",6,1);
EffectEstimate = observed'; CILower = ciLower'; CIUpper = ciUpper';
PValue = p'; TestStatistic = observed'; DF = nan(6,1); Status = repmat("ok",6,1);
result.table = table(Experiment,Condition,Reference,N,EffectMetric,EffectEstimate, ...
    CILower,CIUpper,PValue,TestStatistic,DF,Status);
result.meanPerformance = table(names',mean(participantPerformance,1)', ...
    'VariableNames',{'Condition','MeanPerformance'});
result.participantPerformance = participantPerformance;
result.nullDifferences = nullDifferences;
result.bootstrapDifferences = bootstrapDifferences;
result.seed = 260935;
result.nPermutations = nPermutations;
result.sourceFile = sourceFile;
result.inclusion = sprintf('Experiment statistics use the first %d old participant response rows; later rows are reserved for headphone-test analyses.', ...
    height(metadata));
result.method = "Mean performance difference across all five ratios. Participant bootstrap percentile 95% CI. Two-sided participant permutation with whole-condition sign flips, (b+1)/(B+1); unadjusted p-values. No threshold test.";
result.demographics = struct('NMetadata',height(metadata),'MeanAge',mean(metadata.Age,'omitnan'), ...
    'SDAge',std(metadata.Age,'omitnan'));
result.table.AnalysisSet = repmat("Additional comparison",height(result.table),1);
result.table.TestType = repmat("Participant permutation test",height(result.table),1);
result.table.StatisticName = repmat("Mean performance difference",height(result.table),1);
disp(result.table);
end
