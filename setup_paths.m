function root = setup_paths()
% SETUP_PATHS Add the package's synthesis and utility functions to MATLAB.
root = fileparts(mfilename('fullpath'));
addpath(root, fullfile(root,'synthesis_model'), fullfile(root,'extra_scripts'), ...
    fullfile(root,'observer_models','damping'));
end
