function result = compute_exp1_statistics(nResamples,referenceChoice)
% COMPUTE_EXP1_STATISTICS t-tests, Cohen's d and participant-bootstrap 95% CIs.
% The analysis tests each of the seven discrimination conditions
% against chance (0.5). The original Full-vs-inverted paired t-test and
% Full-vs-chance t-test always run alongside the added comparisons.
% referenceChoice selects added comparisons: "chance" (default) or "full".
if nargin < 1, nResamples = 10000; end
if nargin < 2, referenceChoice = "chance"; end
validateattributes(nResamples, {'numeric'}, {'scalar','integer','positive'});
referenceChoice = string(validatestring(referenceChoice,{'chance','full'}));
folder = fileparts(mfilename('fullpath'));
sourceFile = fullfile(folder,'results_exp1.mat');
data = load(sourceFile,'resp_cond');
responses = double(data.resp_cond);
assert(size(responses,2)==7 && all(ismember(responses(:),[0 1])));
participantPerformance = mean(responses,3);
n = size(responses,1);
names = ["Full generative model","Inverted covariances","2 modes Gaussian", ...
    "Scaled variances","Uniform model","Inverted distributions","Linear decay"];
comparisonConditions = []; referenceIndices = [];
if referenceChoice=="chance"
    comparisonConditions = 1:7; referenceIndices = zeros(1,7);
end
if referenceChoice=="full"
    comparisonConditions = [comparisonConditions 2:7];
    referenceIndices = [referenceIndices ones(1,6)];
end
% Always retain both original tests, with the original subtraction direction.
originalPairs = [1 2; 1 0];
for k = 1:size(originalPairs,1)
    pair = originalPairs(k,:);
    if ~any(comparisonConditions==pair(1) & referenceIndices==pair(2))
        comparisonConditions(end+1) = pair(1); %#ok<AGROW>
        referenceIndices(end+1) = pair(2); %#ok<AGROW>
    end
end
existingMask = ismember([comparisonConditions' referenceIndices'],originalPairs,'rows');
nc = numel(comparisonConditions);
EffectEstimate = zeros(nc,1); PValue = zeros(nc,1);
TestStatistic = zeros(nc,1); DF = zeros(nc,1);
Reference = strings(nc,1); EffectMetric = strings(nc,1);
bootstrapD = zeros(nResamples,nc);
stream = RandStream('mt19937ar','Seed',260901);
% Each index draw is shared by both conditions and all comparisons.
indices = randi(stream,n,[n nResamples]);
for j = 1:nc
    x = participantPerformance(:,comparisonConditions(j));
    if referenceIndices(j)==0
        y = 0.5;
        Reference(j) = "Chance (0.5)";
        EffectMetric(j) = "Cohen d: (Condition - 0.5) / SD(Condition)";
        EffectEstimate(j) = (mean(x)-0.5)/std(x);
        sampled = x(indices);
        bootstrapD(:,j) = ((mean(sampled,1)-0.5)./std(sampled,0,1))';
    else
        y = participantPerformance(:,referenceIndices(j));
        Reference(j) = names(referenceIndices(j));
        % Preserve the original stats file's pooled-marginal-SD convention.
        EffectMetric(j) = "Cohen d_av: (Condition - Reference) / sqrt((SDc^2+SDr^2)/2)";
        EffectEstimate(j) = (mean(x)-mean(y))/sqrt((var(x)+var(y))/2);
        sx = x(indices); sy = y(indices);
        bootstrapD(:,j) = ((mean(sx,1)-mean(sy,1))./sqrt((var(sx,0,1)+var(sy,0,1))/2))';
    end
    [~,PValue(j),~,test] = ttest(x,y);
    TestStatistic(j) = test.tstat; DF(j) = test.df;
end
assert(all(isfinite(bootstrapD),'all'), 'Undefined bootstrap effect size; inspect zero variances.');
ci = prctile(bootstrapD,[2.5 97.5],1);
Experiment = repmat("1",nc,1); Condition = names(comparisonConditions)';
N = repmat(n,nc,1); CILower = ci(1,:)'; CIUpper = ci(2,:)';
Status = repmat("ok",nc,1);
result.table = table(Experiment,Condition,Reference,N,EffectMetric,EffectEstimate, ...
    CILower,CIUpper,PValue,TestStatistic,DF,Status);
result.table.AnalysisSet = repmat("Additional comparison",nc,1);
result.table.AnalysisSet(existingMask) = "Existing comparison; added CI";
result.table.TestType = repmat("Paired t-test",nc,1);
result.table.TestType(referenceIndices==0) = "One-sample t-test";
result.table.StatisticName = repmat("t",nc,1);
result.existingTests = result.table(existingMask,:);
result.ttests = result.table(:,{'Condition','Reference','AnalysisSet','TestType','N','TestStatistic','DF','PValue'});
result.ttests.Properties.VariableNames{'TestStatistic'} = 'TStatistic';
for k = find(existingMask)'
    fprintf('Existing Experiment 1 test: %s vs %s, t(%d) = %.6f, p = %.10g\n', ...
        Condition(k),Reference(k),DF(k),TestStatistic(k),PValue(k));
end
result.participantPerformance = participantPerformance;
result.bootstrapD = bootstrapD;
result.sourceFile = sourceFile;
result.referenceChoice = referenceChoice;
result.seed = 260901;
result.nResamples = nResamples;
result.method = "Two-sided one-sample (chance) or paired (Full) t-test; 95% percentile CI from N-participant bootstrap; unadjusted p-values. Cohen d definition and direction are explicit per row.";
metadata = readtable(fullfile(folder,'..','..','participant_data','Exp1_metadata.csv'));
result.demographics = struct('NMetadata',height(metadata),'MeanAge',mean(metadata.Age,'omitnan'), ...
    'SDAge',std(metadata.Age,'omitnan'));
disp(result.table);
end
