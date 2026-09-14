function SC2(filename,amp,Ci,COrg,depOrg_frac,ws,rhos,rhoo,Z_C3,Z_C4,oN,Vmax,npp,agb_rh,N_Scalar,...
    root_ash,rhizome_ash,Ro_T,Rh_T,bioOrg_frac,max_depth_bgb,tend,k_increase_fast,k_decrease,...
    k_increase_slow,Bref,a_coef,a_depth,b_depth,coef,CO2)
%Function to run a marsh on top of another marsh, for example equilibrate
%to a 1 mm/yr SLRR, and then run for X yrs on top of that at a modern SLRR.
%We use this function to recreate soil chracteristics of the SERC GCREW
%wetlands by building a marsh for ~150 yrs at 3.6 mm/yr SLRR on top of a
%marsh that has equilibrated to 1 mm/yr. 

%Updated 05/2019

%%%----------Start New Model Run----------------------------------------------------------------------------------%%%
tstart=tend+1;%Start of 2nd run
end2=2650;%end of 2st run

R02 = 0.025;%SLRR for new run
RSLRi2 = R02*10^3;

npp_eCO2 = [1.2 1];% proportional change in net carbon uptake [gC/g leaf/day][C3, C4]
agb_rh_eCO2 = [2 1];%[C3, C4] multiplier for rhizome biomass in relation to aboveground biomass
N_Scalar_eCO2=[1 1];%[C3, C4]
eCO2_C3=[1 1.3 1];
eCO2_C4=[1 1 1 1];%
agb_C3=[0 600 0];
agb_C4= [0 500 700 0];

if CO2==1
       npp=npp.*npp_eCO2;
       agb_rh = agb_rh.*agb_rh_eCO2;
       N_Scalar = N_Scalar.*N_Scalar_eCO2;
       coef = [(polyfit((amp-Z_C3),(eCO2_C3.*agb_C3),2)) (polyfit((amp-Z_C4),(eCO2_C4.*agb_C4),3))];
end

files = uigetdir;

depth = struct2array(load([files '/depth.mat']));
d = struct2array(load([files '/d.mat']));
e = struct2array(load([files '/e.mat']));
Z = struct2array(load([files '/Z.mat']));
layer_depth = struct2array(load([files '/layer_depth.mat']));
soilcolumn = struct2array(load([files '/soilcolumn.mat']));

al_fast = struct2array(load([files '/al_fast.mat']));
al_slow = struct2array(load([files '/al_slow.mat']));
bgb_fast = struct2array(load([files '/bgb_fast.mat']));
bgb_slow = struct2array(load([files '/bgb_slow.mat']));
bgb_org_dep = struct2array(load([files '/bgb_org_dep.mat']));
organic = struct2array(load([files '/organic.mat']));
torgacc = struct2array(load([files '/torgacc.mat']));
orgacc = struct2array(load([files '/orgacc.mat']));
org_in = struct2array(load([files '/org_in.mat']));
Fout_C = struct2array(load([files '/Fout_C.mat']));
loi = struct2array(load([files '/loi.mat']));

mineral = struct2array(load([files '/mineral.mat']));
minacc = struct2array(load([files '/minacc.mat']));
accretion = struct2array(load([files '/accretion.mat']));
flooding_dur=struct2array(load([files '/flooding_dur.mat']));
al_min = struct2array(load([files '/al_min.mat']));
bgb_min = struct2array(load([files '/bgb_min.mat']));
species = struct2array(load([files '/species.mat']));
Root = struct2array(load([files '/Root.mat']));
Rhizome = struct2array(load([files '/Rhizome.mat']));
agb = struct2array(load([files '/agb.mat']));
bgb = struct2array(load([files '/bgb.mat']));
spp_weight = struct2array(load([files '/spp_weight.mat']));

column_C = struct2array(load([files '/column_C.mat']));
C_accum = struct2array(load([files '/C_accum.mat']));
perc_C = struct2array(load([files '/perc_C.mat']));
layer_C = struct2array(load([files '/layer_C.mat']));
d_avgk = struct2array(load([files '/d_avgk.mat']));
msl = struct2array(load([files '/msl.mat']));
%msl = struct2array(load([files '/msl_local_hi.mat']));%For IPCC future SLRR scenarios, use as needed
%msl = struct2array(load([files '/msl_local_med.mat']));%For IPCC future SLRR scenarios, use as needed

for ii=(tstart:end2)
msl(ii)=msl(ii-1)+R02;
end

d(tstart:end2) = zeros; e(tstart:end2) = zeros; layer_depth(tstart:end2) = zeros; soilcolumn(tstart:end2) = zeros;
al_min(tstart:end2) = zeros; al_fast(tstart:end2) = zeros; al_slow(tstart:end2) = zeros;
bgb_fast(tstart:end2) = zeros; bgb_slow(tstart:end2) = zeros; bgb_min(tstart:end2) = zeros;
bgb_org_dep(tstart:end2) = zeros; organic(tstart:end2) = zeros; mineral(tstart:end2) = zeros;
orgacc(tstart:end2) = zeros; minacc(tstart:end2) = zeros; org_in(tstart:end2) = zeros;
Fout_C(tstart:end2) = zeros; accretion(tstart:end2) = zeros; dep_alloc_slow(tstart:end2) = zeros;
dep_alloc_fast(tstart:end2) = zeros; dep_alloc_min(tstart:end2) = zeros; perc_C(tstart:end2) = zeros;
layer_C(tstart:end2) = zeros; flooding_dur(tstart:end2) = zeros; C_accum(tstart:end2) = zeros;
column_C(tstart:end2) = zeros; bgb(tstart:end2) = zeros; torgacc(tstart:end2) = zeros;
spp_weight(tstart:end2) = zeros;d_avgk(tstart:end2) = zeros;ktop_fast(tstart:end2) = zeros;
ktop_slow(tstart:end2) = zeros;

outputfilename2 = ['orgRun/' filename '/RSLR_' num2str(RSLRi2) '/Ci_' num2str(Ci) '/'];
if ~exist(outputfilename2, 'dir')
   mkdir(outputfilename2);
end 

for yr = tstart:end2 %% Start time loop
%% Deposition   
    [dep_slow,dep_fast,dep_min,flooding_dur]= deposition(depOrg_frac,COrg,amp,Ci,ws,msl,Z,yr,flooding_dur);

    dep_alloc_slow(yr)= dep_slow;
    dep_alloc_fast(yr)= dep_fast;
    dep_alloc_min(yr)= dep_min;   
%% Biomass
    [tempspecies,temproot,temprhizome,tempagb,kk,d,e,D_min,D_max,Ro_T,Rh_T,dWeight] = biomass(Ro_T,Rh_T,npp,Vmax,agb_rh,oN,N_Scalar,coef,yr,d,e,amp,...
    msl,Z,Z_C3,Z_C4);

    species(yr)=tempspecies;
    Root(yr)=temproot;   
    Rhizome(yr)=temprhizome;
    agb(yr)=tempagb;
    bgb(yr)=temproot+temprhizome;   
    spp_weight(yr) = dWeight;
%% 
ktop_fast(yr) = k_increase_fast*a_coef*(agb(yr)/Bref);%Using calculated coefs - defines k at the top of the soil profile. 
%This number is used to distribute k exponentially down soil column to max. rooting depth
ktop_slow(yr) = k_increase_slow*a_coef*(agb(yr)/Bref);

if ktop_fast(yr) < k_decrease%Doesn't allow ktop to go below the value set in k_decrease 
    ktop_fast(yr) = k_decrease;
end

if ktop_slow(yr) < k_decrease
    ktop_slow(yr) = k_decrease;
end

%% Decompose
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
%% Accretion and Elevation 

accretion(yr)=soilcolumn(yr)-soilcolumn(yr-1);
C_accum(yr)=column_C(yr)-column_C(yr-1);   
Z(yr+1) = Z(yr)+ accretion(yr);          

if yr == end2
    end_yr2 = end2;
    save([outputfilename2 'end_yr2.mat'],'end_yr2');
end

%% Conditions for ending run early
    if d(yr) < min(D_min) 
        end_yr = yr;
        Z(yr) = Z(yr-1);       
        prompt = ['Marsh converted to upland after ', num2str(end_yr),' years'];
        disp(prompt);
%        save([outputfilename2 'end_yr2.mat'],'end_yr2');
        %f_ID(msl,end_yr,R0,amp,tr,Z_start,Z,ws,Ci,CO2,kref_fast,kref_slow,max_depth_bgb,dRdt,tend);
%% Save       
save([outputfilename2 'org_in.mat'],'org_in'); save([outputfilename2 'bgb_org_dep.mat'],'bgb_org_dep'); 
save([outputfilename2 'organic.mat'],'organic');save([outputfilename2 'al_fast.mat'],'al_fast'); 
save([outputfilename2 'al_slow.mat'],'al_slow'); save([outputfilename2 'dep_alloc_slow.mat'],'dep_alloc_slow'); 
save([outputfilename2 'dep_alloc_fast.mat'],'dep_alloc_fast');

save([outputfilename2 'bgb_fast.mat'],'bgb_fast'); save([outputfilename2 'bgb_slow.mat'],'bgb_slow');
save([outputfilename2 'loi.mat'],'loi'); 

save([outputfilename2 'liveroot.mat'],'liveroot'); save([outputfilename2 'liverhizome.mat'],'liverhizome');
save([outputfilename2 'agb.mat'],'agb'); save([outputfilename2 'bgb.mat'],'bgb')
save([outputfilename2 'Root.mat'],'Root'); save([outputfilename2 'Rhizome.mat'],'Rhizome')
save([outputfilename2 'species.mat'],'species');

save([outputfilename2 'Z.mat'],'Z'); save([outputfilename2 'msl.mat'],'msl')
save([outputfilename2 'd.mat'],'d'); save([outputfilename2 'depth.mat'],'depth');
save([outputfilename2 'layer_depth.mat'],'layer_depth');

save([outputfilename2 'flooding_dur.mat'],'flooding_dur');save([outputfilename2 'minacc.mat'],'minacc');
save([outputfilename2 'mineral.mat'],'mineral'); save([outputfilename2 'al_min.mat'],'al_min');
save([outputfilename2 'bgb_min.mat'],'bgb_min'); save([outputfilename2 'dep_alloc_min.mat'],'dep_alloc_min')

save([outputfilename2 'accretion.mat'],'accretion'); save([outputfilename2 'soilcolumn.mat'],'soilcolumn');
save([outputfilename2 'orgacc.mat'],'orgacc');save([outputfilename2 'torgacc.mat'],'torgacc');

save([outputfilename2 'k_slow.mat'],'k_slow'); save([outputfilename2 'k_fast.mat'],'k_fast');
save([outputfilename2 'ktop_fast.mat'],'ktop_fast');
save([outputfilename2 'decomp_bgb_slow.mat'],'decomp_bgb_slow'); save([outputfilename2 'decomp_bgb_fast.mat'],'decomp_bgb_fast');
save([outputfilename2 'decomp_al_slow.mat'],'decomp_al_slow'); save([outputfilename2 'decomp_al_fast.mat'],'decomp_al_fast');
save([outputfilename2 'Fout_C.mat'],'Fout_C');

save([outputfilename2 'layer_C.mat'],'layer_C'); save([outputfilename2 'column_C.mat'],'column_C')
save([outputfilename2 'C_accum.mat'],'C_accum'); save([outputfilename2 'perc_C.mat'],'perc_C')
save([outputfilename2 'spp_weight.mat'],'spp_weight');save([outputfilename2 'd_avgk.mat'],'d_avgk')
save([outputfilename2 'e.mat'],'e');
break

    elseif d(yr) > max(D_max) 
        end_yr = yr;
        Z(yr) = Z(yr-1); 
        prompt = ['Marsh converted to open water after ', num2str(end_yr),' years'];
        disp(prompt);
%         save([outputfilename2 'end_yr2.mat'],'end_yr2');
%% Save       
save([outputfilename2 'org_in.mat'],'org_in'); save([outputfilename2 'bgb_org_dep.mat'],'bgb_org_dep'); 
save([outputfilename2 'organic.mat'],'organic');save([outputfilename2 'al_fast.mat'],'al_fast'); 
save([outputfilename2 'al_slow.mat'],'al_slow'); save([outputfilename2 'dep_alloc_slow.mat'],'dep_alloc_slow'); 
save([outputfilename2 'dep_alloc_fast.mat'],'dep_alloc_fast');

save([outputfilename2 'bgb_fast.mat'],'bgb_fast'); save([outputfilename2 'bgb_slow.mat'],'bgb_slow');
save([outputfilename2 'loi.mat'],'loi'); 

save([outputfilename2 'liveroot.mat'],'liveroot'); save([outputfilename2 'liverhizome.mat'],'liverhizome');
save([outputfilename2 'agb.mat'],'agb'); save([outputfilename2 'bgb.mat'],'bgb')
save([outputfilename2 'Root.mat'],'Root'); save([outputfilename2 'Rhizome.mat'],'Rhizome')
save([outputfilename2 'species.mat'],'species');

save([outputfilename2 'Z.mat'],'Z'); save([outputfilename2 'msl.mat'],'msl')
save([outputfilename2 'd.mat'],'d'); save([outputfilename2 'depth.mat'],'depth');
save([outputfilename2 'layer_depth.mat'],'layer_depth');

save([outputfilename2 'flooding_dur.mat'],'flooding_dur');save([outputfilename2 'minacc.mat'],'minacc');
save([outputfilename2 'mineral.mat'],'mineral'); save([outputfilename2 'al_min.mat'],'al_min');
save([outputfilename2 'bgb_min.mat'],'bgb_min'); save([outputfilename2 'dep_alloc_min.mat'],'dep_alloc_min')

save([outputfilename2 'accretion.mat'],'accretion'); save([outputfilename2 'soilcolumn.mat'],'soilcolumn');
save([outputfilename2 'orgacc.mat'],'orgacc');save([outputfilename2 'torgacc.mat'],'torgacc');

save([outputfilename2 'k_slow.mat'],'k_slow'); save([outputfilename2 'k_fast.mat'],'k_fast');
save([outputfilename2 'ktop_fast.mat'],'ktop_fast');
save([outputfilename2 'decomp_bgb_slow.mat'],'decomp_bgb_slow'); save([outputfilename2 'decomp_bgb_fast.mat'],'decomp_bgb_fast');
save([outputfilename2 'decomp_al_slow.mat'],'decomp_al_slow'); save([outputfilename2 'decomp_al_fast.mat'],'decomp_al_fast');
save([outputfilename2 'Fout_C.mat'],'Fout_C');

save([outputfilename2 'layer_C.mat'],'layer_C'); save([outputfilename2 'column_C.mat'],'column_C')
save([outputfilename2 'C_accum.mat'],'C_accum'); save([outputfilename2 'perc_C.mat'],'perc_C')
save([outputfilename2 'spp_weight.mat'],'spp_weight');save([outputfilename2 'd_avgk.mat'],'d_avgk')
save([outputfilename2 'e.mat'],'e');
break
    end

end
%% Save files

save([outputfilename2 'org_in.mat'],'org_in'); save([outputfilename2 'bgb_org_dep.mat'],'bgb_org_dep'); 
save([outputfilename2 'organic.mat'],'organic');save([outputfilename2 'al_fast.mat'],'al_fast'); 
save([outputfilename2 'al_slow.mat'],'al_slow'); save([outputfilename2 'dep_alloc_slow.mat'],'dep_alloc_slow'); 
save([outputfilename2 'dep_alloc_fast.mat'],'dep_alloc_fast');

save([outputfilename2 'bgb_fast.mat'],'bgb_fast'); save([outputfilename2 'bgb_slow.mat'],'bgb_slow');
save([outputfilename2 'loi.mat'],'loi'); 

save([outputfilename2 'liveroot.mat'],'liveroot'); save([outputfilename2 'liverhizome.mat'],'liverhizome');
save([outputfilename2 'agb.mat'],'agb'); save([outputfilename2 'bgb.mat'],'bgb')
save([outputfilename2 'Root.mat'],'Root'); save([outputfilename2 'Rhizome.mat'],'Rhizome')
save([outputfilename2 'species.mat'],'species');

save([outputfilename2 'Z.mat'],'Z'); save([outputfilename2 'msl.mat'],'msl')
save([outputfilename2 'd.mat'],'d'); save([outputfilename2 'depth.mat'],'depth');
save([outputfilename2 'layer_depth.mat'],'layer_depth');

save([outputfilename2 'flooding_dur.mat'],'flooding_dur');save([outputfilename2 'minacc.mat'],'minacc');
save([outputfilename2 'mineral.mat'],'mineral'); save([outputfilename2 'al_min.mat'],'al_min');
save([outputfilename2 'bgb_min.mat'],'bgb_min'); save([outputfilename2 'dep_alloc_min.mat'],'dep_alloc_min')

save([outputfilename2 'accretion.mat'],'accretion'); save([outputfilename2 'soilcolumn.mat'],'soilcolumn');
save([outputfilename2 'orgacc.mat'],'orgacc');save([outputfilename2 'torgacc.mat'],'torgacc');

save([outputfilename2 'k_slow.mat'],'k_slow'); save([outputfilename2 'k_fast.mat'],'k_fast');
save([outputfilename2 'ktop_fast.mat'],'ktop_fast');
save([outputfilename2 'decomp_bgb_slow.mat'],'decomp_bgb_slow'); save([outputfilename2 'decomp_bgb_fast.mat'],'decomp_bgb_fast');
save([outputfilename2 'decomp_al_slow.mat'],'decomp_al_slow'); save([outputfilename2 'decomp_al_fast.mat'],'decomp_al_fast');
save([outputfilename2 'Fout_C.mat'],'Fout_C');

save([outputfilename2 'layer_C.mat'],'layer_C'); save([outputfilename2 'column_C.mat'],'column_C')
save([outputfilename2 'C_accum.mat'],'C_accum'); save([outputfilename2 'perc_C.mat'],'perc_C')
save([outputfilename2 'spp_weight.mat'],'spp_weight');save([outputfilename2 'd_avgk.mat'],'d_avgk')
save([outputfilename2 'e.mat'],'e');

%%
if yr == end2 
    prompt = ['Marsh remains after ', num2str(end_yr2),' years'];
    disp(prompt); 
end