function SMART_COHORT(filename)%(filename,RO) if running for multiple user input SLR rates. (filename) if running for SLR rate specified in code.
%SMART_COHORT

%A dynamic point-based soil-cohort model for the morphological evolution of a marsh soil column.
%The model captures the ecogeomorphic feedbacks between flooding, organic matter accumulation, sediment deposition,
%and marsh surface elevation under scenarios of sea level rise and elevated CO2. Changes in primary productivity
%and sedimentation are based elevation above MSL, and the decay rate of organic matter scales with aboveground biomass
%and exponentially declines with depth. The model incorporates the differential responses of C3 and C4 plants, allowing
%for the switching between parameterizations for each plant type, creating weighted mixed communities as defined by the
%overlapping ranges of each species along an elevation continuum, and incorporating the empirically derived relationship
%between C3 productivity and elevated CO2.

%Model developed by Anthony Rietl, Ellen Herbert, and Matthew Kirwan (kirwan@vims.edu). Contact Kirwan with questions.

%Updated 05/2019. This version used in Rietl et al., in prep.

%% Model Parameters
CO2=0;%ambient = 0; elevated = 1
tend=2500;%end of 1st run

R0=0.011;
RSLRi = R0*10^3;%[m/yr] Initial rate of sea level rise; RSLRi is [mm/yr]
tr = .44; amp = tr/2;%[m] tidal range & amplitude

Ci=5;%[mg/L] concentrtion of suspended sediemnt (mineral + organic)
COrg = 0.05;%[g/g] organic content of suspended seidment Ci
depOrg_frac = [0.1 0.9];%fast and slow sediment OM fractions %mlk- is this specific to organic matter in supsended sediment, where 10% is fast pool, 90% slow pool?
ws = 2e-4;%[m/s]
rhos= 1.99e6; rhoo= 8.5e4;%[g/m3] bulk density of mineral matter Morris et al. 2016;[g/m3] bulk density of organic matter Morris et al. 2016

%Vegetation growth vs elevaiton parameters. The model fits a biomass curve through a known set of biomass (agb_C3) and elevation (Z_C3) points.
agb_C3=[0 600 0];%[g/m2] Abovegroud biomass of C3 species (S. americanus) across elevation gradient coresponding to elevations in Z_C3
Z_C3=[0.28 0.16 0.05];%[m_NAVD88] Elevation for biomass of C3 species from line above above.
eCO2_C3=[1 1.3 1];%C3 Scirpus increase in biomass in response to elevated based on Chunwu Zhu's observations at SERC (e.g. ~20% increase in agb at optimum elevation)
%(Manuscript in review:After 30 years, rising sea level negates carbon dioxide-induced stimulation of plant production in a coastal wetland megonigalp@si.edu for data)

agb_C4= [0 500 700 0];%[g/m2] Abovegroud biomass of C4 species (S. patens) across elevation gradient coresponding to elevations in Z_C4
Z_C4=[0.45 0.38 0.27 0.12];%[m_NAVD88] Elevation for biomass of C4 species (Patens) from line above above.
eCO2_C4=[1 1 1 1];% C4 does not increase biomass in response to eCO2.
  
%Parameters for root:shoot calculations and eCO2 scaling
npp=[0.175 0.175];%[C3, C4] proportional change in net carbon uptake [gC/g leaf/day]
agb_rh=[1 1];%[C3, C4] multiplier for rhizome biomass in relation to aboveground biomass
N_Scalar=[.6 .6];

npp_eCO2 = [1.2 1];% proportional change in net carbon uptake [gC/g leaf/day][C3, C4]
agb_rh_eCO2 = [2 1];%[C3, C4] multiplier for rhizome biomass in relation to aboveground biomass
N_Scalar_eCO2=[1 1];%[C3, C4]

Vmax= [3.26e-3 5.3e-3];%[C3, C4] maximum Nitrogen uptake velocity
oN=[.02 .02 .02 .02];%Optimal [N] in tissue (fraction) C3, C4, and mixed communities [C3, C4, C3 Mixed Dominant (C3MD), C4 Mixed Dominant (C4MD)]

%Fits the rlationship between biomass and elevation for ambient or eCO2 conditions
coef = [(polyfit((amp-Z_C3),agb_C3,2)) (polyfit((amp-Z_C4),agb_C4,3))];

    if CO2==1
       npp=npp.*npp_eCO2;
       agb_rh = agb_rh.*agb_rh_eCO2;
       N_Scalar = N_Scalar.*N_Scalar_eCO2;
       coef = [(polyfit((amp-Z_C3),(eCO2_C3.*agb_C3),2)) (polyfit((amp-Z_C4),(eCO2_C4.*agb_C4),3))];
    end

%Belowground parameters
root_ash=[.08 .08 .08 .08];%Fraction of root biomass that is organic [C3, C4, C3MD, C4MD]
rhizome_ash=[.08 .08 .08 .08];%Fraction of rhizome biomass that is organic [C3, C4, C3MD, C4MD]
Ro_T = [1.2 1.6];%[yr] Multiplier for belowground biomass - number of times root biomass diesback and regrows
Rh_T = [0.6 1]; %[yr] Multiplier for belowground biomass - number of times rhizome biomass diesback and regrows
bioOrg_frac = [0.84 0.16];% fast and slow sediment OM fractions
max_depth_bgb = 1;%[m] maximum Rooting depth

%Decomposition parameters - Creates a linear relationship between decomposition rate and aboveground biomass
kref_fast = 0.027;% [yr^-1] Reference/average decomposition constant of labile organic matter
kref_slow = 0.0027;% [yr^-1] Reference/average decomposition constant of recalcitrant organic matter

k_decrease = 0.0001;%Defines the lowest decomposition rate (i.e. decomposition is never = 0)
x = linspace(127,900,1000)';%Range of biomass values to fit linear relationship - lowest biomass (127) defined by Mueller et al 2015.
%When biomass is below 127, k = k_decrease
Bref = 500;%[g/m^2/yr] Reference/average aboveground biomass in a given year - i.e. the biomass at which you find
%your kref value

    k_increase_fast = kref_fast*2;%The maximum amount by which kref can increase - doubling estimated from Mueller et al 2015
    k_increase_slow = kref_slow*2;
    kFast_range = linspace(k_decrease,k_increase_fast,1000)';%Full range of k from k_decrease to k_increase (lowest to highest possible values)

    f_fast = fit(x,kFast_range,'0.054*a_coef*(x/500)','StartPoint',0.1);%fits the relationship between range of k values and aboveground biomass
    %the fit function doesnt allow for defined variables thus must be entered manually - 0.048 is k_increase_fast, a_coef will be calculated,
    %x is the range in biomass values, 500 is Bref
    a_coef = coeffvalues(f_fast);%Coef of the relationship, used in calculating k for a given aboveground biomass value

    %This section calculated coefs for distributing k with depth in the soil column
    x2 = fliplr(linspace(0,1,1000))';%Defines 1000 points between 0 and 1 - refers to soil column and max. rooting depth, i.e. 0 to 1 m
    f_depth = fit( x2,kFast_range,'0.054*a*exp(b*x)','StartPoint',[0.1 -0.1]);%Fits the range of k values to depth - exponential distribution to max. rooting depth
    coef_depth = coeffvalues(f_depth);
    a_depth = coef_depth(1);%coef values used to calculate k at given depth
    b_depth = coef_depth(2);

%Allows you to run a marsh on top of a preexisting marsh. Comment out through elseif str == 'n', and prompt at end if automating several runs at different SLRRs
prompt = 'Do you have pesVars? (y/n) ';
str = input(prompt,'s');
if str == 'y'
    SC2(filename,amp,Ci,COrg,depOrg_frac,ws,rhos,rhoo,Z_C3,Z_C4,oN,Vmax,npp,agb_rh,N_Scalar,...
    root_ash,rhizome_ash,Ro_T,Rh_T,bioOrg_frac,max_depth_bgb,tend,k_increase_fast,k_decrease,...
    k_increase_slow,Bref,a_coef,a_depth,b_depth,coef,CO2)
    return
elseif str == 'n'
%% Preallocate variables for Pre Existing Stratigraphy
msl=zeros(1,tend);
for i=1000:tend%creates msl matrix relative to your R0
    msl(i)= msl(i-1)+R0;
end

layer_depth= ones(1,1000)*0.001;%Defines thickness of each layer - 0.001 m/1 mm

%Preallocate
accretion=zeros(1,tend); minacc=zeros(1,tend); soilcolumn = zeros(1,tend); Z=zeros(1,tend);

al_min1=ones(1,1000)*(rhos*.001);% mineral mass in each layer equivalent to a layer depth of 1 mm
al_min2= zeros(1,tend-1000);
al_min= horzcat(al_min1,al_min2);

Z_end = 0.25;%Starting elevation (m) - user defined to begin marsh building whereever you want in vegetqation growth range

Z_start=Z_end-(1000*.001);%To build the stratigraphy, Z increment must be same as layer_depth - default is 1000 layers, each 1 mm thick.
for i=1:1000
   Z(i)=Z_start+(i*.001);
   soilcolumn(i)=Z(i)-Z_start;
   if i > 1
   minacc(i) = Z(i) - Z(i-1);
   accretion(i) = Z(i) - Z(i-1);
   end
end
%Preallocation
bgb_org_dep= zeros(1,tend); bgb_fast= zeros(1,tend); bgb_slow= zeros(1,tend); bgb_min= zeros(1,tend);
al_fast= zeros(1,tend); al_slow= zeros(1,tend); layer_C= zeros(1,tend); loi= zeros(1,tend); organic = zeros(1,tend);
mineral = zeros(1,tend); dep_alloc_slow=zeros(1,tend); dep_alloc_fast=zeros(1,tend); dep_alloc_min=zeros(1,tend);
species=zeros(1,tend); Root=zeros(1,tend); Rhizome=zeros(1,tend); agb=zeros(1,tend); bgb=zeros(1,tend);
Fout_C=zeros(1,tend); column_C=zeros(1,tend); C_accum=zeros(1,tend); d=zeros(1,tend); e=zeros(1,tend); torgacc=zeros(1,tend);
orgacc=zeros(1,tend); org_in=zeros(1,tend);flooding_dur=zeros(1,tend); spp_weight=zeros(1,tend);d_avgk = zeros(1,tend);
ktop_fast=zeros(1,tend); ktop_slow=zeros(1,tend);
%% Name of the folder where outputs/plots will be saved
outputfilename = ['pesVars/' filename '/RSLR_' num2str(RSLRi) '/Ci_' num2str(Ci) '/'];
if ~exist(outputfilename, 'dir')%if directory doesn't exist, create one
   mkdir(outputfilename);
end

%% Model Loop
for yr = 1000:tend
%% Deposition - Function to calculate the amount of sediment trapping and settling for a tide range and suspended sediment concentration
    [dep_slow,dep_fast,dep_min,flooding_dur]= deposition(depOrg_frac,COrg,amp,Ci,ws,msl,Z,yr,flooding_dur);

    dep_alloc_slow(yr)= dep_slow;%Allocthonous slow organic deposition
    dep_alloc_fast(yr)= dep_fast;%Allocthonous fast organic deposition
    dep_alloc_min(yr)= dep_min;%Allocthonous mineral deposition
%% Biomass - Function to calculate biomass related chracteristics
    [tempspecies,temproot,temprhizome,tempagb,kk,d,e,D_min,D_max,Ro_T,Rh_T,dWeight] = biomass(Ro_T,Rh_T,npp,Vmax,agb_rh,oN,N_Scalar,coef,yr,d,e,amp,...
    msl,Z,Z_C3,Z_C4);

    species(yr)=tempspecies;%1=C3, 2=C4, 3=C3MD, 4=C4MD
    Root(yr)=temproot;%[g/m^2/yr] Root biomass in a given year
    Rhizome(yr)=temprhizome;%[g/m^2/yr] Rhizome biomass in a given year
    agb(yr)=tempagb;%[g/m^2/yr] Aboveground biomass in a given year
    bgb(yr)=temproot+temprhizome;%[g/m^2/yr] Belowground biomass in a given year
    spp_weight(yr) = dWeight;%Weight (fraction of total) in mixed species communities. Number always refers to the dominant species
%% Decomposition - calculating decay rate at soil surface based on aboveground biomass

ktop_fast(yr) = k_increase_fast*a_coef*(agb(yr)/Bref);%Using calculated coefs - defines k at the top of the soil profile.
%This number is used to distribute k exponentially down soil column to max. rooting depth
ktop_slow(yr) = k_increase_slow*a_coef*(agb(yr)/Bref);

if ktop_fast(yr) < k_decrease%Doesn't allow ktop to go below the value set in k_decrease
    ktop_fast(yr) = k_decrease;
end

if ktop_slow(yr) < k_decrease
    ktop_slow(yr) = k_decrease;
end

%% Decompose - Function to calculate all organic matter distribution and decomposition related parameters
   [tempdavgk,tempFout_C,tempsoilcolumn,tempcolumn_C,temporgacc,tempminacc,temporgin,k_slow,k_fast,decomp_al_fast,decomp_al_slow,decomp_bgb_fast,...
    decomp_bgb_slow,bgb_org_dep,depth,liveroot,liverhizome,al_fast,al_slow,al_min,bgb_fast,bgb_slow,bgb_min,layer_depth,layer_C,loi,organic,...
    mineral,perc_C] = decompose(bioOrg_frac,max_depth_bgb,kk,temproot,temprhizome,dep_slow,dep_fast,...
    dep_min,yr,rhos,rhoo,al_fast,al_slow,al_min,bgb_fast,bgb_slow,bgb_min,layer_depth,layer_C,loi,organic,mineral,bgb_org_dep,Ro_T,Rh_T,root_ash,...
    rhizome_ash,ktop_fast,ktop_slow,a_depth,b_depth);

    Fout_C(yr)=tempFout_C;%[g/m^2/yr] Total decomposition of organic matter
    soilcolumn(yr)=tempsoilcolumn;%[m] Depth of soil column as measured from bottom up (largest value at surface)
    column_C(yr)=tempcolumn_C;%[g/m^2/yr] Total Carbon in soil profile
    torgacc(yr) = temporgacc;%only used in orgacc calc.
    minacc(yr) = tempminacc;%[m] Mineral accretion
    org_in(yr) = temporgin;%[g/m^2/yr] Total organic matter inputs
    orgacc(yr) = torgacc(yr)- torgacc(yr-1);%[m] Organic accretion
    d_avgk(yr) = tempdavgk;%Depth averaged decay rate
%% Accretion and Elevation - Calculates change in elevation
accretion(yr)=soilcolumn(yr)-soilcolumn(yr-1);
C_accum(yr)=column_C(yr)-column_C(yr-1);%[g/m^2/yr] Carbon accumulation rate
Z(yr+1) = Z(yr)+ accretion(yr);

if yr == tend
    end_yr = tend;
    save([outputfilename 'end_yr.mat'],'end_yr');
end
%% Conditions for ending run early - Upland conversion
     if d(yr) < min(D_min)
        end_yr = yr;
        Z(yr) = Z(yr-1);
        prompt = ['Marsh converted to upland after ', num2str(end_yr),' years'];
        disp(prompt);
        save([outputfilename 'end_yr.mat'],'end_yr');
%% Save
save([outputfilename 'org_in.mat'],'org_in'); save([outputfilename 'bgb_org_dep.mat'],'bgb_org_dep');
save([outputfilename 'organic.mat'],'organic');save([outputfilename 'al_fast.mat'],'al_fast');
save([outputfilename 'al_slow.mat'],'al_slow'); save([outputfilename 'dep_alloc_slow.mat'],'dep_alloc_slow');
save([outputfilename 'dep_alloc_fast.mat'],'dep_alloc_fast');

save([outputfilename 'bgb_fast.mat'],'bgb_fast'); save([outputfilename 'bgb_slow.mat'],'bgb_slow');
save([outputfilename 'loi.mat'],'loi');

save([outputfilename 'liveroot.mat'],'liveroot'); save([outputfilename 'liverhizome.mat'],'liverhizome');
save([outputfilename 'agb.mat'],'agb'); save([outputfilename 'bgb.mat'],'bgb')
save([outputfilename 'Root.mat'],'Root'); save([outputfilename 'Rhizome.mat'],'Rhizome')
save([outputfilename 'species.mat'],'species');

save([outputfilename 'Z.mat'],'Z'); save([outputfilename 'msl.mat'],'msl')
save([outputfilename 'd.mat'],'d'); save([outputfilename 'depth.mat'],'depth');
save([outputfilename 'layer_depth.mat'],'layer_depth');

save([outputfilename 'flooding_dur.mat'],'flooding_dur');save([outputfilename 'minacc.mat'],'minacc');
save([outputfilename 'mineral.mat'],'mineral'); save([outputfilename 'al_min.mat'],'al_min');
save([outputfilename 'bgb_min.mat'],'bgb_min'); save([outputfilename 'dep_alloc_min.mat'],'dep_alloc_min')

save([outputfilename 'accretion.mat'],'accretion'); save([outputfilename 'soilcolumn.mat'],'soilcolumn');
save([outputfilename 'orgacc.mat'],'orgacc'); save([outputfilename 'torgacc.mat'],'torgacc');

save([outputfilename 'k_slow.mat'],'k_slow'); save([outputfilename 'k_fast.mat'],'k_fast');
save([outputfilename 'ktop_fast.mat'],'ktop_fast');
save([outputfilename 'decomp_bgb_slow.mat'],'decomp_bgb_slow'); save([outputfilename 'decomp_bgb_fast.mat'],'decomp_bgb_fast');
save([outputfilename 'decomp_al_slow.mat'],'decomp_al_slow'); save([outputfilename 'decomp_al_fast.mat'],'decomp_al_fast');
save([outputfilename 'Fout_C.mat'],'Fout_C');

save([outputfilename 'layer_C.mat'],'layer_C'); save([outputfilename 'column_C.mat'],'column_C')
save([outputfilename 'C_accum.mat'],'C_accum'); save([outputfilename 'perc_C.mat'],'perc_C')

save([outputfilename 'spp_weight.mat'],'spp_weight');save([outputfilename 'd_avgk.mat'],'d_avgk')
save([outputfilename 'e.mat'],'e');
break
 
%% Conditions for ending run early - Open water conversion
    elseif d(yr) > max(D_max)
        end_yr = yr;
        Z(yr) = Z(yr-1);
        prompt = ['Marsh converted to open water after ', num2str(end_yr),' years'];
        disp(prompt);
        save([outputfilename 'end_yr.mat'],'end_yr');
%% Save
save([outputfilename 'org_in.mat'],'org_in'); save([outputfilename 'bgb_org_dep.mat'],'bgb_org_dep');
save([outputfilename 'organic.mat'],'organic');save([outputfilename 'al_fast.mat'],'al_fast');
save([outputfilename 'al_slow.mat'],'al_slow'); save([outputfilename 'dep_alloc_slow.mat'],'dep_alloc_slow');
save([outputfilename 'dep_alloc_fast.mat'],'dep_alloc_fast');

save([outputfilename 'bgb_fast.mat'],'bgb_fast'); save([outputfilename 'bgb_slow.mat'],'bgb_slow');
save([outputfilename 'loi.mat'],'loi');

save([outputfilename 'liveroot.mat'],'liveroot'); save([outputfilename 'liverhizome.mat'],'liverhizome');
save([outputfilename 'agb.mat'],'agb'); save([outputfilename 'bgb.mat'],'bgb')
save([outputfilename 'Root.mat'],'Root'); save([outputfilename 'Rhizome.mat'],'Rhizome')
save([outputfilename 'species.mat'],'species');

save([outputfilename 'Z.mat'],'Z'); save([outputfilename 'msl.mat'],'msl')
save([outputfilename 'd.mat'],'d'); save([outputfilename 'depth.mat'],'depth');
save([outputfilename 'layer_depth.mat'],'layer_depth');

save([outputfilename 'flooding_dur.mat'],'flooding_dur');save([outputfilename 'minacc.mat'],'minacc');
save([outputfilename 'mineral.mat'],'mineral'); save([outputfilename 'al_min.mat'],'al_min');
save([outputfilename 'bgb_min.mat'],'bgb_min'); save([outputfilename 'dep_alloc_min.mat'],'dep_alloc_min')

save([outputfilename 'accretion.mat'],'accretion'); save([outputfilename 'soilcolumn.mat'],'soilcolumn');
save([outputfilename 'orgacc.mat'],'orgacc'); save([outputfilename 'torgacc.mat'],'torgacc');

save([outputfilename 'k_slow.mat'],'k_slow'); save([outputfilename 'k_fast.mat'],'k_fast');
save([outputfilename 'ktop_fast.mat'],'ktop_fast');
save([outputfilename 'decomp_bgb_slow.mat'],'decomp_bgb_slow'); save([outputfilename 'decomp_bgb_fast.mat'],'decomp_bgb_fast');
save([outputfilename 'decomp_al_slow.mat'],'decomp_al_slow'); save([outputfilename 'decomp_al_fast.mat'],'decomp_al_fast');
save([outputfilename 'Fout_C.mat'],'Fout_C');

save([outputfilename 'layer_C.mat'],'layer_C'); save([outputfilename 'column_C.mat'],'column_C')
save([outputfilename 'C_accum.mat'],'C_accum'); save([outputfilename 'perc_C.mat'],'perc_C')
save([outputfilename 'spp_weight.mat'],'spp_weight');save([outputfilename 'd_avgk.mat'],'d_avgk')
save([outputfilename 'e.mat'],'e');
break
    end

end

%% Save files
save([outputfilename 'org_in.mat'],'org_in'); save([outputfilename 'bgb_org_dep.mat'],'bgb_org_dep');
save([outputfilename 'organic.mat'],'organic');save([outputfilename 'al_fast.mat'],'al_fast');
save([outputfilename 'al_slow.mat'],'al_slow'); save([outputfilename 'dep_alloc_slow.mat'],'dep_alloc_slow');
save([outputfilename 'dep_alloc_fast.mat'],'dep_alloc_fast')

save([outputfilename 'bgb_fast.mat'],'bgb_fast'); save([outputfilename 'bgb_slow.mat'],'bgb_slow');
save([outputfilename 'loi.mat'],'loi');

save([outputfilename 'liveroot.mat'],'liveroot'); save([outputfilename 'liverhizome.mat'],'liverhizome');
save([outputfilename 'agb.mat'],'agb'); save([outputfilename 'bgb.mat'],'bgb')
save([outputfilename 'Root.mat'],'Root'); save([outputfilename 'Rhizome.mat'],'Rhizome')
save([outputfilename 'species.mat'],'species');

save([outputfilename 'Z.mat'],'Z'); save([outputfilename 'msl.mat'],'msl')
save([outputfilename 'd.mat'],'d'); save([outputfilename 'depth.mat'],'depth');
save([outputfilename 'layer_depth.mat'],'layer_depth');

save([outputfilename 'flooding_dur.mat'],'flooding_dur');save([outputfilename 'minacc.mat'],'minacc');
save([outputfilename 'mineral.mat'],'mineral'); save([outputfilename 'al_min.mat'],'al_min');
save([outputfilename 'bgb_min.mat'],'bgb_min'); save([outputfilename 'dep_alloc_min.mat'],'dep_alloc_min')

save([outputfilename 'accretion.mat'],'accretion'); save([outputfilename 'soilcolumn.mat'],'soilcolumn');
save([outputfilename 'orgacc.mat'],'orgacc'); save([outputfilename 'torgacc.mat'],'torgacc');

save([outputfilename 'k_slow.mat'],'k_slow'); save([outputfilename 'k_fast.mat'],'k_fast');
save([outputfilename 'ktop_fast.mat'],'ktop_fast');
save([outputfilename 'decomp_bgb_slow.mat'],'decomp_bgb_slow'); save([outputfilename 'decomp_bgb_fast.mat'],'decomp_bgb_fast');
save([outputfilename 'decomp_al_slow.mat'],'decomp_al_slow'); save([outputfilename 'decomp_al_fast.mat'],'decomp_al_fast');
save([outputfilename 'Fout_C.mat'],'Fout_C');

save([outputfilename 'layer_C.mat'],'layer_C'); save([outputfilename 'column_C.mat'],'column_C')
save([outputfilename 'C_accum.mat'],'C_accum'); save([outputfilename 'perc_C.mat'],'perc_C')
save([outputfilename 'spp_weight.mat'],'spp_weight');save([outputfilename 'd_avgk.mat'],'d_avgk')
save([outputfilename 'e.mat'],'e');

if yr == tend
    prompt = ['Marsh remains after ', num2str(end_yr),' years'];
    disp(prompt);
end

end
% Comment out this prompt through to the end following "elseif str == 'n'" if running batch runs through multiple SLRRs
prompt = 'Run a new marsh on top? (y/n) ';
str = input(prompt,'s');
if str == 'y'
    SC2(filename,amp,Ci,COrg,depOrg_frac,ws,rhos,rhoo,Z_C3,Z_C4,oN,Vmax,npp,agb_rh,N_Scalar,...
    root_ash,rhizome_ash,Ro_T,Rh_T,bioOrg_frac,max_depth_bgb,tend,k_increase_fast,k_decrease,...
    k_increase_slow,Bref,a_coef,a_depth,b_depth,coef,CO2)
    return
elseif str == 'n'
end

end


