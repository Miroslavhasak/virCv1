clc; clear; close all;

%% --- Bludisko 16x16 (0=cesta, 1=stena) ---
maze = [
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1;
1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1;
1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1;
1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1;
1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1;
1 0 0 0 0 0 0 0 1 0 0 0 1 0 0 1;
1 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1;
1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1;
1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1;
1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1;
1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1;
1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1;
1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1;
1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1;
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1;
];

%% --- Parametre ---
pocet_generacii = 200;
pocet_jedncov    = 50;
hidden_neurony   = 10;
input_size       = 11; % 3x3 okolie + delta_x, delta_y
output_size      = 4;  % hore, dole, vlavo, vpravo
start_goal       = [1 1 8 11]; % start a ciel

%% --- Inicializacia populacie ---
n_wagi = input_size*hidden_neurony + hidden_neurony*output_size + hidden_neurony + output_size;
populacia = randn(pocet_jedncov, n_wagi);

%% --- Trening GA ---
for gen = 1:pocet_generacii
    Fit = zeros(pocet_jedncov,1);
    for i = 1:pocet_jedncov
        Fit(i) = mlp_fitness(populacia(i,:), maze, start_goal(1:2), start_goal(3:4), input_size, hidden_neurony, output_size);
    end

    [~, idx] = sort(Fit);
    Best  = populacia(idx(1:round(0.2*pocet_jedncov)),:);
    Old   = populacia(idx(round(0.2*pocet_jedncov)+1:round(0.4*pocet_jedncov)),:);
    Work1 = populacia(idx(round(0.4*pocet_jedncov)+1:round(0.7*pocet_jedncov)),:);
    Work2 = populacia(idx(round(0.7*pocet_jedncov)+1:end),:);

    Work1 = crossov_MLP(Work1, 0.7);
    Work2 = mutx_MLP(Work2, 0.1);
    populacia = [Best; Old; Work1; Work2];
end

disp('Trening dokoncený!');

%% --- Simulacia a vykreslenie konecnej cesty ---
best_weights = populacia(1,:);
path_pos = simulate_mlp(best_weights, maze, start_goal(1:2), start_goal(3:4), input_size, hidden_neurony, output_size, 200);

figure;
imagesc(~maze); colormap(gray); axis equal tight; axis ij; hold on;
plot(start_goal(2), start_goal(1), 'go','MarkerFaceColor','g'); % start
plot(start_goal(4), start_goal(3), 'ro','MarkerFaceColor','r'); % ciel

% Vykreslenie zelenej cesty
if size(path_pos,1) > 1
    plot(path_pos(:,2), path_pos(:,1),'g.-','LineWidth',2,'MarkerSize',8);
end
plot(path_pos(end,2), path_pos(end,1),'co','MarkerFaceColor','c','MarkerSize',10); % cyan bod
title('Najlepsia cesta po treningu');

%% --- Funkcie ---
function fitness = mlp_fitness(vahy, maze, start_pos, goal_pos, input_size, hidden_neurony, output_size)
    max_steps = 50; pos = start_pos; visited=zeros(size(maze));
    idx = 1;
    W1 = reshape(vahy(idx:idx+input_size*hidden_neurony-1), [hidden_neurony,input_size]); idx=idx+input_size*hidden_neurony;
    W2 = reshape(vahy(idx:idx+hidden_neurony*output_size-1), [output_size,hidden_neurony]); idx=idx+hidden_neurony*output_size;
    b1 = vahy(idx:idx+hidden_neurony-1); idx=idx+hidden_neurony;
    b2 = vahy(idx:idx+output_size-1);
    
    fitness = 0;
    for t=1:max_steps
        input = get_local_input(maze,pos,goal_pos);
        h = tanh(W1*input' + b1(:));
        o = W2*h + b2(:);
        [~,move] = max(o);
        next_pos = pos;
        switch move
            case 1, next_pos(1)=pos(1)-1;
            case 2, next_pos(1)=pos(1)+1;
            case 3, next_pos(2)=pos(2)-1;
            case 4, next_pos(2)=pos(2)+1;
        end
        if next_pos(1)<1||next_pos(1)>size(maze,1)||next_pos(2)<1||next_pos(2)>size(maze,2)||maze(next_pos(1),next_pos(2))==1
            fitness = fitness + 5;
        else
            pos = next_pos;
            if visited(pos(1),pos(2))==1, fitness=fitness+1; end
            visited(pos(1),pos(2))=1;
        end
        fitness = fitness + abs(pos(1)-goal_pos(1)) + abs(pos(2)-goal_pos(2));
        if pos(1)==goal_pos(1) && pos(2)==goal_pos(2), break; end
    end
end

function input = get_local_input(maze,pos,goal_pos)
    r = pos(1); c = pos(2); [m,n] = size(maze);
    neighborhood = zeros(3,3);
    for i=-1:1
        for j=-1:1
            rr = r+i; cc = c+j;
            if rr<1||rr>m||cc<1||cc>n
                neighborhood(i+2,j+2) = 1;
            else
                neighborhood(i+2,j+2) = maze(rr,cc);
            end
        end
    end
    delta = goal_pos-pos;
    input = [neighborhood(:)' delta./max(size(maze))]; % 9 steny + 2 delta
end

function novy=crossov_MLP(pop,p)
    [n,m]=size(pop); novy=pop;
    for i=1:2:n-1
        if rand<p
            cp=randi(m-1);
            temp=novy(i,cp+1:end); novy(i,cp+1:end)=novy(i+1,cp+1:end);
            novy(i+1,cp+1:end)=temp;
        end
    end
end

function novy=mutx_MLP(pop,p)
    [n,m]=size(pop); novy=pop;
    for i=1:n
        for j=1:m
            if rand<p
                novy(i,j)=novy(i,j)+randn*0.5;
            end
        end
    end
end

function path_pos = simulate_mlp(vahy, maze, start_pos, goal_pos, input_size, hidden_neurony, output_size, max_steps)
    pos = start_pos; path_pos = pos;
    idx = 1;
    W1 = reshape(vahy(idx:idx+input_size*hidden_neurony-1), [hidden_neurony,input_size]); idx=idx+input_size*hidden_neurony;
    W2 = reshape(vahy(idx:idx+hidden_neurony*output_size-1), [output_size,hidden_neurony]); idx=idx+hidden_neurony*output_size;
    b1 = vahy(idx:idx+hidden_neurony-1); idx=idx+hidden_neurony;
    b2 = vahy(idx:idx+output_size-1);
    
    for t=1:max_steps
        input = get_local_input(maze,pos,goal_pos);
        h = tanh(W1*input'+b1(:));
        o = W2*h + b2(:);
        [~,move] = max(o);
        next_pos = pos;
        switch move
            case 1, next_pos(1)=pos(1)-1;
            case 2, next_pos(1)=pos(1)+1;
            case 3, next_pos(2)=pos(2)-1;
            case 4, next_pos(2)=pos(2)+1;
        end
        if next_pos(1)<1||next_pos(1)>size(maze,1)||next_pos(2)<1||next_pos(2)>size(maze,2)||maze(next_pos(1),next_pos(2))==1
            break;
        end
        pos = next_pos;
        path_pos = [path_pos; pos];
        if pos(1)==goal_pos(1) && pos(2)==goal_pos(2), break; end
    end
end
