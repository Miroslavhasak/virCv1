clc; clear; close all;

% Bludisko 16x16 (0 = cesta, 1 = stena)
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

% Parametre
pocet_generacii = 500;
pocet_jedncov = 50;
hodnota_v_jedincovi = 10; % Dĺžka cesty (počet krokov)
pocet_behov = 10;

start_pos = [1, 1];  % Štartová pozícia
goal_pos = [16, 11];  % Cieľová pozícia

% Hlavný cyklus
for int = 1:pocet_behov
    %% Inicializácia populácie
    Space = [ones(1, hodnota_v_jedincovi); 4*ones(1, hodnota_v_jedincovi)];
    populacia = randi([1, 4], pocet_jedncov, hodnota_v_jedincovi); % Generovanie počiatočnej populácie

    %% Hlavný cyklus generácií
    for gen = 1:pocet_generacii
        % Funkcia fitness
        Fit = testfn_maze(populacia, start_pos, goal_pos, maze);  % NOVÁ fitness funkcia

        % Genetický algoritmus
        Best = selbest(populacia, Fit, [4,4]);  
        Old = selsus(populacia, Fit, 8);       
        Work1 = selsus(populacia, Fit, 14);
        Work2 = selsus(populacia, Fit, 20);

        % Kríženie a mutácie
        Work1 = crossov(Work1, 1, 0); 
        Work2 = mutx(Work2, 0.15, Space);  

        % Spojenie generácií
        populacia = [Best; Old; Work1; Work2];
    end
end

% Vizualizácia bludiska
figure(2);
imshow(maze, 'InitialMagnification', 'fit');
hold on;
plot(start_pos(2), start_pos(1), 'go', 'MarkerFaceColor', 'g'); % Štart
plot(goal_pos(2), goal_pos(1), 'ro', 'MarkerFaceColor', 'r');   % Cieľ

% Vykreslenie najlepšej cesty
best_path = populacia(1, :);  
path_pos = start_pos;
for step = 1:length(best_path)
    switch best_path(step)
        case 1, path_pos(2) = path_pos(2) + 1; % Doprava
        case 2, path_pos(1) = path_pos(1) + 1; % Dole
        case 3, path_pos(2) = path_pos(2) - 1; % Doľava
        case 4, path_pos(1) = path_pos(1) - 1; % Hore
    end
    plot(path_pos(2), path_pos(1), 'b.', 'MarkerSize', 10);
end
hold off;

%% NOVÁ fitness funkcia pre bludisko
function fitness = testfn_maze(populacia, start_pos, goal_pos, maze)
    nJedincov = size(populacia, 1);
    fitness = zeros(nJedincov,1);
    for i = 1:nJedincov
        pos = start_pos;
        penalizacia = 0;
        visited = zeros(size(maze));
        cesta = populacia(i,:);
        for step = 1:length(cesta)
            move = cesta(step);
            next_pos = pos;
            switch move
                case 1, next_pos(2) = pos(2) + 1; % Doprava
                case 2, next_pos(1) = pos(1) + 1; % Dole
                case 3, next_pos(2) = pos(2) - 1; % Doľava
                case 4, next_pos(1) = pos(1) - 1; % Hore
            end
            % Kontrola hraníc a stien
            if next_pos(1) < 1 || next_pos(1) > size(maze,1) || ...
               next_pos(2) < 1 || next_pos(2) > size(maze,2) || ...
               maze(next_pos(1), next_pos(2)) == 1
                penalizacia = penalizacia + 10;
            else
                pos = next_pos;
                if visited(pos(1), pos(2)) == 1
                    penalizacia = penalizacia + 5; % penalizacia za návrat
                end
                visited(pos(1), pos(2)) = 1;
            end
            % Cieľ
            if pos(1)==goal_pos(1) && pos(2)==goal_pos(2)
                fitness(i) = 0;
                break;
            end
        end
        % Fitness = Manhattan + penalizácia
        if fitness(i) ~= 0
            fitness(i) = abs(pos(1)-goal_pos(1)) + abs(pos(2)-goal_pos(2)) + penalizacia;
        end
    end
end
