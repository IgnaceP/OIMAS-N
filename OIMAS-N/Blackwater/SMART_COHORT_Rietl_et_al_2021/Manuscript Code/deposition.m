function [dep_slow,dep_fast,dep_min,flooding_dur]= deposition(depOrg_frac,COrg,amp,Ci,ws,msl,Z,yr,flooding_dur)

%Function to calculate the amount of sediment trapping and settling for a
%tide range and suspended sediment concentration (C, mg/l OR g/m3)1.
%D’Alpaos A, Lanzoni S, Marani M, Rinaldo A (2007) Landscape evolution in tidal 
%embayments: Modeling the interplay of erosion, sedimentation, and vegetation 
%dynamics. J Geophys Res Earth Surf 112:1–17

%Updated 04/2019

%% Function Parameters
numiterations = 500;
P=12.5*3600; % tidal period [s]
dt = P/numiterations;
timestep=365*(24/12.5); %[tidal cycles per year] number to multiply accretion simulated over a tidal cycle by

%Initiate variables for saving through time for one tidal cycle
C=zeros(1,numiterations);
qs=zeros(1,numiterations);
depth=zeros(1,numiterations);
%% Iterate through a single tidal cycle
for t = 2:numiterations
    depth(t)=amp*sin(2*pi*(t*dt/P-.25))+(msl(yr)-Z(yr)); %Depth of water over marsh surface, as a function of tidal stage
        if depth(t) < 0
            depth(t) = 0; %If tide is out, depth is zero
        elseif depth(t) > 0
            flooding_dur(yr) = flooding_dur(yr) + dt; 
            %note:ws*Ci*flooding_dur = est. of sed deposited in tidal cycle
            %ws*Ci*Flood_dur by number of tidal cycles in a yr to get
            %annual
            dh = depth(t)-depth(t-1); %Change in the water level (above marsh surface) from previous time step
            if dh > 0
                mi = C(t-1)*depth(t-1); %initial mass of sediment in the water column above the marsh platform
                dm = -qs(t-1)+Ci*dh; %change in mass - mass balance between deposition and sediment coming in with tide (SSC)
                if dm+mi < 0 %Cannot remove more sediment from the water column than is there
                    dm=-mi; %To keep concentration from becoming negative, set it to zero in this case
                end
                C(t) = (mi+dm)/(depth(t)); %new suspended sediment concentration is equal to the initial mass of sediment plus the change in mass of sediment, divided by the depth of the water column
                qs(t) = C(t)*ws*dt; %sediment settling (Marani et al., 2007)
            else
                mi = C(t-1).*depth(t-1); %initial mass of sediment in the water column above the marsh platform
                dm = -qs(t-1)+C(t-1)*dh; %change in mass - mass balance between deposition and sediment going out with tide
                if dm+mi < 0 %Cannot remove more sediment from the water column than is there
                    dm=-mi; %To keep concentration from becoming negative, set it to zero in this case
                end
                C(t) = (mi+dm)/(depth(t)); %new suspended sediment concentration is equal to the initial mass of sediment plus the change in mass of sediment, divided by the depth of the water column
                            
                qs(t) = C(t)*ws*dt; %sediment settling (Marani et al., 2007)

            end
        end
end
%% Calculate annual depostion rates from trapping and settling

dep=sum(qs)*timestep; %Annual settling mass flux (g/m2/yr)
dep_slow = COrg*dep*depOrg_frac(1);
dep_fast = COrg*dep*depOrg_frac(2);
dep_min = dep - (dep_slow+dep_fast);
if abs(dep -(dep_slow + dep_fast + dep_min))<1
    prompt_run = ['Running...yr ',num2str(yr)];
    disp (prompt_run)
else
    disp 'check dep calc'
end