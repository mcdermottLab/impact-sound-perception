function result = compute_exp2_statistics(nResamples)
% COMPUTE_EXP2_STATISTICS Recorded-confusion correlations, paired permutations and CIs.
% Resample N participants together across Recorded, Full and all alternatives.
% Permutations exchange the two complete condition matrices within participant.
% Pearson correlations use all 49 confusion-matrix cells, as in the original.
if nargin < 1, nResamples = 10000; end
validateattributes(nResamples, {'numeric'}, {'scalar','integer','positive'});
folder = fileparts(mfilename('fullpath'));
sourceFile = fullfile(folder, 'Expt2_data.mat');
data = load(sourceFile, 'resp_fin', 'key_part', 'matnum_part');
responses = double(data.resp_fin);
keys = double(data.key_part(:));
materials = double(data.matnum_part(:));
assert(size(responses,2) == numel(keys) && numel(keys) == numel(materials));
assert(all(ismember(responses(:),1:7)) && all(ismember(materials,1:7)));
assert(isequal(unique(keys)',1:8), 'Expected eight conditions.');
names = ["Recorded", "Full generative model", "Inverted covariances", ...
    "Uniform distribution", "Two modes Gaussian", "Two modes uniform", ...
    "Linear decay Gaussian", "Linear decay uniform"];
n = size(responses,1);
counts = zeros(n,8,49);
for participant = 1:n
    for condition = 1:8
        mask = keys == condition;
        cm = accumarray([materials(mask), responses(participant,mask)'],1,[7 7]);
        counts(participant,condition,:) = cm(:);
    end
end
assert(all(sum(counts,3) == 28,'all'), 'Expected 28 trials per condition.');
total = reshape(sum(counts,1),8,49);
correlations = row_correlations(total(1,:),total);
observed = correlations(2) - correlations(3:8);
stream = RandStream('mt19937ar','Seed',260902);
bootstrapDifferences = zeros(nResamples,6);
nullDifferences = zeros(nResamples,6);
for b = 1:nResamples
    indices = randi(stream,n,[n 1]);
    bootTotal = reshape(sum(counts(indices,:,:),1),8,49);
    bootCorr = row_correlations(bootTotal(1,:),bootTotal);
    bootstrapDifferences(b,:) = bootCorr(2)-bootCorr(3:8);
    for j = 1:6
        pair = counts(:,[2 j+2],:);
        swap = rand(stream,n,1) < 0.5;
        pair(swap,:,:) = pair(swap,[2 1],:);
        permTotal = reshape(sum(pair,1),2,49);
        permCorr = row_correlations(total(1,:),permTotal);
        nullDifferences(b,j) = permCorr(1)-permCorr(2);
    end
end
assert(all(isfinite(bootstrapDifferences),'all') && all(isfinite(nullDifferences),'all'), ...
    'A confusion correlation is undefined; do not silently discard draws.');
ci = prctile(bootstrapDifferences,[2.5 97.5],1);
p = (1+sum(abs(nullDifferences) >= abs(observed)-1e-12,1))/(nResamples+1);
Experiment = repmat("2",6,1); Condition = names(3:8)';
Reference = repmat("Full generative model",6,1); N = repmat(n,6,1);
EffectMetric = repmat("r(Recorded,Full) - r(Recorded,Condition)",6,1);
EffectEstimate = observed'; CILower = ci(1,:)'; CIUpper = ci(2,:)';
PValue = p'; TestStatistic = observed'; DF = nan(6,1);
Status = repmat("ok",6,1);
result.table = table(Experiment,Condition,Reference,N,EffectMetric,EffectEstimate, ...
    CILower,CIUpper,PValue,TestStatistic,DF,Status);
result.conditionCorrelations = table(names',correlations', ...
    'VariableNames',{'Condition','CorrelationWithRecorded'});
result.bootstrapDifferences = bootstrapDifferences;
result.nullDifferences = nullDifferences;
result.confusionCounts = counts;
result.sourceFile = sourceFile;
result.seed = 260902;
result.nResamples = nResamples;
result.method = "Paired participant bootstrap; 95% percentile CI; two-sided participant permutation, (b+1)/(B+1); unadjusted p-values.";
metadata = readtable(fullfile(folder,'..','..','participant_data','Exp2_metadata.csv'));
result.demographics = struct('NMetadata',height(metadata),'MeanAge',mean(metadata.Age,'omitnan'), ...
    'SDAge',std(metadata.Age,'omitnan'));
result.table.AnalysisSet = repmat("Additional comparison",height(result.table),1);
result.table.TestType = repmat("Participant permutation test",height(result.table),1);
result.table.StatisticName = repmat("Correlation difference",height(result.table),1);
result.table.AnalysisSet(1) = "Existing comparison; added CI";
result.existingTests = result.table(1,:);
disp(result.table);
end

function r = row_correlations(reference,rows)
reference = reference-mean(reference);
rows = rows-mean(rows,2);
r = (rows*reference')' ./ sqrt(sum(rows.^2,2)'*sum(reference.^2));
end
