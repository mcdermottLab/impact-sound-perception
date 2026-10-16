function run_paper_figures(pythonExecutable)
% RUN_PAPER_FIGURES Recreate the main figure panels from any working folder.
% run_paper_figures('python3') uses the Python environment containing the
% packages in requirements.txt. Figure 6 performs a lengthy Monte Carlo run.
if nargin < 1, pythonExecutable = 'python3'; end
root = setup_paths();
previousFolder = pwd;
cleanupFolder = onCleanup(@() cd(previousFolder)); %#ok<NASGU>
mkdir(fullfile(root,'Figures','all_figs'));
entries = {
    'Figure 2','plot_figure_02d_power_explained'; 'Figure 2','plot_figure_02e_fit_quality'; 'Figure 2','plot_figure_02h_modal_parameter_distributions';
    'Figure 3','plot_figure_03ab_modes_by_material'; 'Figure 3','plot_figure_03cd_modes_by_size';
    'Figure 4','plot_figure_04_force_comparison'; 'Figure 4','plot_figure_04f_amplitude_by_mass';
    'Figure 6','plot_figure_06a_model_likelihoods';
    'Figure 7','plot_figure_07a_discrimination'; 'Figure 7','plot_figure_07bc_material_confusions';
    'Figure 8','plot_figure_08abc_mass_discrimination_exp3a'; 'Figure 8','plot_figure_08def_mass_discrimination_exp3b';
    'Figure 9','plot_figure_09a_amplitude_observer'; 'Figure 9','plot_figure_09b_normalized_amplitude_observer';
    'Figure 9','plot_figure_09c_centroid_observer';
    'Figure 10','plot_figure_10acd_damping_discrimination'; 'Figure 10','plot_figure_10b_damping_parameters'};
for k = 1:size(entries,1)
    fprintf('Running %s/%s\n',entries{k,1},entries{k,2});
    cd(fullfile(root,'Figures',entries{k,1}));
    execute_script(entries{k,2});
end
cd(fullfile(root,'Figures','Figure 5'));
[status, output] = system(sprintf('"%s" plot_figure_05_material_confusions.py',pythonExecutable));
fprintf('%s',output);
assert(status == 0,'Figure 5 Python script failed. Check requirements.txt.');
end

function execute_script(name)
% Each original figure script uses clearvars. Give it a separate workspace.
run(name);
end
