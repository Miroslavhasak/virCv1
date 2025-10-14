% neuroevo_autonom.m
clc; clear; close all;

% --- Definicia bludiska 16x16 (0=cesta, 1=stena) ---
maze = [
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
    0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0;
    0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0;
    0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0;
    0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0;
    0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0;
    0 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0;
    0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0;
    0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0;
    0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0;
    0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0;
    0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0;
    0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0;
    0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0;
    0 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0;
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
];

% --- Parametre trenovania a sieti ---
pocet_generacii = 300;     % pocet generacii
pocet_jedncov    = 60;     % velkost populacie
hidden_neurony   = 12;     % skryta vrstva
input_size       = 4;      % x, y, x_goal, y_goal (normalizovane)
output_size      = 4;      % smer (prava, dole, lava, hore)
max_steps        = 80;     % pocet krokov pri simulacii

% geneticke nastavenia
elit_frac        = 0.20;   % podiel elitnych jedincov
cross_p          = 0.75;   % pravdepodobnost krizenia v bloku Work1
mut_p            = 0.15;   % pravdepodobnost mutacie (per-genome element)
rand_starts_per_gen = 2;   % kolko nahodnych startov pridat do fitnessu kazdu generaciu

% vizualizacia počas trenovania
anim_interval = 10;        % kazdu X generaciu vykresli best jedinca
demo_start = [1,1];        % start pre demo animaciu
demo_goal  = [8,11];      % ciel pre demo animaciu
fixed_goal = demo_goal;    % fixny ciel pre random starty v treningu

% Treningove start-ciel dvojice (fixne scenare)
start_goal_pairs = [
    1 1 1 11;
    1 1 16 16;
    1 16 16 1;
];

% inicializacia populacie (nahodne vahy)
n_wagi = input_size*hidden_neurony + hidden_neurony*output_size + hidden_neurony + output_size;
populacia = randn(pocet_jedncov, n_wagi);

% ak chces zistit statistik y fitnessu
best_history = zeros(pocet_generacii,1);
mean_history = zeros(pocet_generacii,1);

% hlavny evolucny cyklus
for gen = 1:pocet_generacii
    Fit = zeros(pocet_jedncov,1);
    for i = 1:pocet_jedncov
        fitness_i = 0;
        % hodnotenie na fixnych scenaroch
        for pair = 1:size(start_goal_pairs,1)
            start_pos = start_goal_pairs(pair,1:2);
            goal_pos  = start_goal_pairs(pair,3:4);
            fitness_i = fitness_i + mlp_fitness(populacia(i,:), input_size, hidden_neurony, output_size, maze, start_pos, goal_pos, max_steps);
        end
        % hodnotenie na nahodnych startoch (pomaha generalizacii)
        for r = 1:rand_starts_per_gen
            start_pos = random_free_cell(maze);
            goal_pos  = fixed_goal;
            fitness_i = fitness_i + mlp_fitness(populacia(i,:), input_size, hidden_neurony, output_size, maze, start_pos, goal_pos, max_steps);
        end
        Fit(i) = fitness_i;
    end

    % zaznamenaj statistiky
    [sortedFit, idx] = sort(Fit);
    best_history(gen) = sortedFit(1);
    mean_history(gen) = mean(Fit);

    % selekcia
    Best  = populacia(idx(1:round(elit_frac*pocet_jedncov)),:);
    Old   = populacia(idx(round(elit_frac*pocet_jedncov)+1:round(2*elit_frac*pocet_jedncov)),:);
    Work1 = populacia(idx(round(2*elit_frac*pocet_jedncov)+1:round(0.7*pocet_jedncov)),:);
    Work2 = populacia(idx(round(0.7*pocet_jedncov)+1:end),:);

    % gen. operatory
    Work1 = crossov_MLP(Work1, cross_p);
    Work2 = mutx_MLP(Work2, mut_p);

    % nova populacia
    populacia = [Best; Old; Work1; Work2];

    % animacia najlepsieho jedinca
    if mod(gen, anim_interval) == 0 || gen==1 || gen==pocet_generacii
        best_weights = populacia(1,:);
        visualize_path(best_weights, maze, demo_start, demo_goal, input_size, hidden_neurony, output_size, max_steps, gen);
        % jednoduchy graf fitnessu podmaze
        figure(2); clf;
        plot(1:gen, mean_history(1:gen), '.-'); hold on;
        plot(1:gen, best_history(1:gen), '.-'); hold off;
        xlabel('generacia'); ylabel('fitness'); legend('mean','best'); title('Fitness priebeh');
        drawnow;
    end
end

% finalne testovanie - mozes menit startovnu poziciu a spustit cast znova
best_weights = populacia(1,:);
test_starts = [1 1; 16 16; 1 16; 4 13; 16 1]; % priklad roznych startov
figure(3); clf;
for s = 1:size(test_starts,1)
    st = test_starts(s,:);
    visualize_path(best_weights, maze, st, demo_goal, input_size, hidden_neurony, output_size, max_steps, s);
    pause(0.6);
end

% ------------------ FUNKCIE ------------------

function fitness = mlp_fitness(vahy, input_size, hidden_neurony, output_size, maze, start_pos, goal_pos, max_steps)
    % simuluj vozidlo riadene MLP od start_pos po max_steps krokoch
    pos = start_pos;
    visited = zeros(size(maze));
    idx = 1;
    W1 = reshape(vahy(idx:idx+input_size*hidden_neurony-1), [hidden_neurony, input_size]); idx = idx+input_size*hidden_neurony;
    W2 = reshape(vahy(idx:idx+hidden_neurony*output_size-1), [output_size, hidden_neurony]); idx = idx+hidden_neurony*output_size;
    b1 = vahy(idx:idx+hidden_neurony-1); idx = idx+hidden_neurony;
    b2 = vahy(idx:idx+output_size-1);

    fitness = 0;
    for t = 1:max_steps
        input = [pos ./ size(maze,1), goal_pos ./ size(maze,1)]; % normalizovane vstupy
        h = tanh(W1 * input' + b1(:));
        o = W2 * h + b2(:);
        [~, move] = max(o); % vyber akcie s najvacsim outputom

        next_pos = pos;
        switch move
            case 1, next_pos(2) = pos(2) + 1; % doprava
            case 2, next_pos(1) = pos(1) + 1; % dole
            case 3, next_pos(2) = pos(2) - 1; % dolava
            case 4, next_pos(1) = pos(1) - 1; % hore
        end

        % ak kolizia alebo mimo mapy => penalizuj a pokracuj dalej (zastav tah)
        if next_pos(1)<1 || next_pos(1)>size(maze,1) || ...
           next_pos(2)<1 || next_pos(2)>size(maze,2) || ...
           maze(next_pos(1), next_pos(2))==1
            fitness = fitness + 10; % penalizacia za naraz
            % nepovazuj tento krok za posun
            continue;
        end

        pos = next_pos;
        if visited(pos(1), pos(2))==1
            fitness = fitness + 2; % penalizuj opakovane navstevy
        end
        visited(pos(1), pos(2)) = 1;

        % pridaj vzdialenost k cielu (manhattanova)
        dist = abs(pos(1)-goal_pos(1)) + abs(pos(2)-goal_pos(2));
        fitness = fitness + dist;

        if dist == 0
            fitness = fitness - 100; % velka odmena za dosiahnutie ciela
            break;
        end
    end
end

function novy = crossov_MLP(pop, p)
    % jednoduchy jednopunktovy blokovy crossover (v ramci parov)
    [n, m] = size(pop);
    novy = pop;
    for i = 1:2:n-1
        if rand < p
            cp = randi(m-1);
            temp = novy(i,cp+1:end);
            novy(i,cp+1:end) = novy(i+1,cp+1:end);
            novy(i+1,cp+1:end) = temp;
        end
    end
end

function novy = mutx_MLP(pop, p)
    % mutacia: pre kazdy genotypny prvok s pravdepodobnostou p pridaj gaussianu
    [n, m] = size(pop);
    novy = pop;
    sigma = 0.3;
    for i = 1:n
        for j = 1:m
            if rand < p
                novy(i,j) = novy(i,j) + randn*sigma;
            end
        end
    end
end

function visualize_path(weights, maze, start_pos, goal_pos, input_size, hidden_neurony, output_size, max_steps, gen)
    % vykresli cestu jedinca v mape; pouziva rovnaku logiku ako fitness
    pos = start_pos;
    path = pos;
    idx = 1;
    W1 = reshape(weights(idx:idx+input_size*hidden_neurony-1), [hidden_neurony, input_size]); idx = idx+input_size*hidden_neurony;
    W2 = reshape(weights(idx:idx+hidden_neurony*output_size-1), [output_size, hidden_neurony]); idx = idx+hidden_neurony*output_size;
    b1 = weights(idx:idx+hidden_neurony-1); idx = idx+hidden_neurony;
    b2 = weights(idx:idx+output_size-1);

    for t = 1:max_steps
        input = [pos ./ size(maze,1), goal_pos ./ size(maze,1)];
        h = tanh(W1 * input' + b1(:));
        o = W2 * h + b2(:);
        [~, move] = max(o);
        next_pos = pos;
        switch move
            case 1, next_pos(2) = pos(2) + 1;
            case 2, next_pos(1) = pos(1) + 1;
            case 3, next_pos(2) = pos(2) - 1;
            case 4, next_pos(1) = pos(1) - 1;
        end

        if next_pos(1)<1 || next_pos(1)>size(maze,1) || ...
           next_pos(2)<1 || next_pos(2)>size(maze,2) || ...
           maze(next_pos(1), next_pos(2))==1
            break;
        end

        pos = next_pos;
        path = [path; pos];
        if pos(1)==goal_pos(1) && pos(2)==goal_pos(2)
            break;
        end
    end

    figure(1);
    imshow(maze, 'InitialMagnification','fit');
    hold on;
    plot(start_pos(2), start_pos(1), 'go', 'MarkerFaceColor','g');
    plot(goal_pos(2), goal_pos(1), 'ro', 'MarkerFaceColor','r');
    plot(path(:,2), path(:,1), 'c.-', 'LineWidth',1.5);
    title(['Generacia / test: ', num2str(gen)]);
    hold off;
    drawnow;
end

function cell = random_free_cell(maze)
    % vrati nahodnu volnu (0) bunku v maze (okrem okrajov)
    [r,c] = find(maze==0);
    % ignoruj okrajove volne bunky ak chces, momentalne berieme vsetky
    idx = randi(length(r));
    cell = [r(idx), c(idx)];
end
