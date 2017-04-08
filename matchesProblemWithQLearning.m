%% Initialization
gamma=1;
alpha = 3*1e-1;   % suggested: 3*1e-1
epsilon = 0.2;  % for our epsilon greedy policy, suggested: 0.2
nStates = 20; 
nActions = 2; 
MAX_N_EPISODES=1e3; %suggested: 1e5 with epsilon = 0.01 and alpha = 3*1e-1 for precision of 0.01.
Q = zeros(nStates,nActions);
% keep track of how many timestep we take per episode:
ets = zeros(MAX_N_EPISODES,1); ts=0; 
for ei=1:MAX_N_EPISODES,
  tic; 
  if( ei==1 ) 
    fprintf('working on episode %d...\n',ei);
  else
    fprintf('working on episode %d (ptt=%10.6f secs)...\n',ei, toc); tic; 
  end
  ets(ei,1) = ts+1; 
  % initialize the starting state
  st = nStates; 

  % pick action using an epsilon greedy policy derived from Q: 
  [dum,at] = max(Q(st,:));  % at \in [1,2,3,4]=[up,down,right,left]
  if( rand<epsilon )         % explore ... with a random action 
    tmp=randperm(nActions); at=tmp(1); 
  end
  
  % begin the episode
  while( 1 ) 
    ts=ts+1; 
    %fprintf('st=(%d,%d); act=%3d...\n',st(1),st(2),at);
    % propagate to state stp1 and collect a reward rew
    [rew,stp1] = step(st,at,nActions); 
    % make the greedy action selection: 
    [dum,atp1] = max(Q(stp1,:)); 
    if( rand<epsilon )         % explore ... with a random action 
      tmp=randperm(nActions); atp1=tmp(1); 
    end
    if( ~( stp1==1) ) % stp1 is not the terminal state
      Q(st,at) = Q(st,at) + alpha*( rew + gamma*max(Q(stp1,:)) - Q(st,at) );
      
    else                                                  % stp1 IS the terminal state ... no Q(s';a') term in the sarsa update
      Q(st,at) = Q(st,at) + alpha*( rew - Q(st,at) );
      
      break; 
    end
    st = stp1; at = atp1; 
    
  end % end policy while 
end % end episode loop 

display(Q);
function [rew,stp1] = step(st,act,nActions)
if (st==2)
  stp1=1;
  if (act==1)
    rew=1;
  else
    rew=0;
  end;
elseif (st==3)
  stp1=1;
  if (act==2)
    rew=1;
  else
    rew=0;
  end;
else
ran=randperm(nActions);
opponent_action=ran(1);
  switch act
   case 1, 
    stp1 = max(st-1-opponent_action,1);
    rew=0;
   case 2,
    stp1 = max(st-2-opponent_action,1);
    rew=0;
  end;
end;

end


