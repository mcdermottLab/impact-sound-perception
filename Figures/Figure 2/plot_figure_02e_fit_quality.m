close all;
clearvars

load('fit_results.mat')
% Create errorbar plot with lines
figure;
set(gcf, 'Units', 'inches', 'Position', [2, 2, 4, 4]);
avg_gof = [avg_gof_constant, avg_gof_linear, avg_gof_quadratic, avg_gof_cubic];
var_gof = [var_gof_constant, var_gof_linear, var_gof_quadratic, var_gof_cubic];

errorbar(1:4, avg_gof, sqrt(var_gof), '-o', 'LineWidth', 2, 'MarkerSize', 5, 'CapSize', 0,"Color",[0 0 0], 'MarkerFaceColor', 'k');
hold on
% Add individual data points with jitter
jitter_width = 0.2;
scatter(1 + jitter_width*(rand(size(gof_constant))-0.5), gof_constant, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.05);
scatter(2 + jitter_width*(rand(size(gof_linear))-0.5), gof_linear, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.05);
scatter(3 + jitter_width*(rand(size(gof_quadratic))-0.5), gof_quadratic, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.05);
scatter(4 + jitter_width*(rand(size(gof_cubic))-0.5), gof_cubic, 10, 'k', 'filled', 'MarkerFaceAlpha', 0.05);
% Redraw errorbar on top
errorbar(1:4, avg_gof, sqrt(var_gof), '-o', 'LineWidth', 2, 'MarkerSize', 5, 'CapSize', 0, "Color",[0 0 0], 'MarkerFaceColor', 'k');

% Customize the plot
xlabel('Polynomial Fit Type');
ylabel('Average R-squared');
xticks(1:4)
xticklabels({'Constant', 'Linear', 'Quadratic', 'Cubic'});
ylim([0, 1.15]);
xlim([-0.5 4.5])
set(gca, 'FontSize', 12);

% Save the figure
saveas(gcf,'../all_figs/Figure_2e.svg');

function [fitresult0, fitresult1, fitresult2, fitresult3, gof0, gof1, gof2, gof3] = create_polynomial_fits(x, y)
    % Remove any Inf or NaN values
    valid = isfinite(x) & isfinite(y);
    x = x(valid);
    y = y(valid);
    
    % Ensure x and y are column vectors
    x = x(:);
    y = y(:);
    
    % Check if we have enough valid data points
    if length(x) < 4  % We need at least 4 points for a 3rd degree polynomial
        fitresult0 = []; fitresult1 = []; fitresult2 = []; fitresult3 = [];
        gof0 = []; gof1 = []; gof2 = []; gof3 = [];
        warning('Insufficient data points for polynomial fit.');
        return;
    end
    
    x_fit = x;
    y_fit = y;
    
    % Check again if we have enough data points after selecting up to intersection point
    if length(x_fit) < 4
        fitresult0 = []; fitresult1 = []; fitresult2 = []; fitresult3 = [];
        gof0 = []; gof1 = []; gof2 = []; gof3 = [];
        warning('Insufficient data points for polynomial fit after intersection point selection.');
        return;
    end

     % Fit constant (custom implementation)
     constant_fit = mean(y_fit);
     fitresult0 = @(x) constant_fit * ones(size(x));
     residuals0 = y_fit - fitresult0(x_fit);
     SSE0 = sum(residuals0.^2);
     SST0 = sum((y_fit - mean(y_fit)).^2);
     R20 = 1 - SSE0/SST0;
     MAE0 = mean(abs(residuals0));
     gof0 = struct('sse', SSE0, 'rsquare', R20, 'mae', MAE0);
    
    % Fit constant, 1st, 2nd, and 3rd degree polynomials
    [fitresult1, gof1] = fit(x_fit, y_fit, 'poly1');
    [fitresult2, gof2] = fit(x_fit, y_fit, 'poly2');
    [fitresult3, gof3] = fit(x_fit, y_fit, 'poly3');
    
    % Calculate MAE for polynomial fits
    gof1.mae = mean(abs(y_fit - fitresult1(x_fit)));
    gof2.mae = mean(abs(y_fit - fitresult2(x_fit)));
    gof3.mae = mean(abs(y_fit - fitresult3(x_fit)));
end
