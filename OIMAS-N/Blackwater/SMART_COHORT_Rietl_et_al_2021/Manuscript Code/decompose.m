function [tempdavgk,tempFout_C,tempsoilcolumn,tempcolumn_C,temporgacc,tempminacc,temporgin,k_slow,k_fast,decomp_al_fast,decomp_al_slow,decomp_bgb_fast,...
       decomp_bgb_slow,bgb_org_dep,depth,liveroot,liverhizome,al_fast,al_slow,al_min,bgb_fast,bgb_slow,bgb_min,layer_depth,layer_C,loi,organic,...
       mineral,perc_C] = decompose(bioOrg_frac,max_depth_bgb,kk,temproot,temprhizome,dep_slow,dep_fast,...
       dep_min,yr,rhos,rhoo,al_fast,al_slow,al_min,bgb_fast,bgb_slow,bgb_min,layer_depth,layer_C,loi,organic,mineral,bgb_org_dep,Ro_T,Rh_T,root_ash,...
       rhizome_ash,ktop_fast,ktop_slow,a_depth,b_depth)
   %Function to calculate all decomposition related parameters. Distributes
   %live roots exponentially down the soil column, calculates OM in each
   %cohort, includes total turnover and depth dependent decay rates 
   
   %Updated 05/2019
%% Function Parameters
g= 0.27;%The depth at which belowground biomass is reduced by 1/3 - Mudd et al 2009
bro=temproot/g;
brh=temprhizome/g;
layer_depth(yr)=layer_depth(yr-1);%Layer depth must be defined before calculating true layer depth, thus approximated by last years value 

%% Add and decompose material in each layer (cohort) of the soil column  
for cohort = yr:-1:1 %Loop through each pocket of sediment in each cell, starting at the surface
    
depth(cohort) = sum(layer_depth(cohort+1:yr));
    if depth(cohort) < max_depth_bgb  

    topdepth=depth(cohort); 
    bottomdepth=depth(cohort)+layer_depth(cohort); 

    fun1=@(depth) bro.*exp(-depth/g); %Defines the function that distributes biomass exponentially with depth in soil profile         
    fun2=@(depth) brh.*exp(-depth/g);

    liveroot(cohort)=integral(fun1,topdepth,bottomdepth,'ArrayValued',true);%Integrates under the depth distribution curve for each cohort to calculate biomass
    liverhizome(cohort)=integral(fun2,topdepth,bottomdepth,'ArrayValued',true);

    liveroot_Turnover(cohort) = liveroot(cohort).*Ro_T(kk);
    liverhizome_Turnover(cohort) = liverhizome(cohort).*Rh_T(kk);
    total_Turnover(cohort) = liveroot_Turnover(cohort) + liverhizome_Turnover(cohort);%Total Turnover in the soil (root + rhizome)  

    root_min_dep(cohort) = liveroot_Turnover(cohort)*root_ash(kk); 
    rhizome_min_dep(cohort) = liverhizome_Turnover(cohort)*rhizome_ash(kk);
    tempbgb_min(cohort)=root_min_dep(cohort)+rhizome_min_dep(cohort);%Mineral deposistion in the soil column from fraction of biomass thats mineral

    bgb_org_dep(cohort)= total_Turnover(cohort) - tempbgb_min(cohort);%Calculates total organic inputs - turnover minus fraction thats mineral      
    tempbgb_fast(cohort) = bgb_org_dep(cohort)*bioOrg_frac(1);%Splits total into fast and slow fractions (labile and recalcitrant) 
    tempbgb_slow(cohort) = bgb_org_dep(cohort)*bioOrg_frac(2);

%If the cohort is on the surface, allocthonous material from suspedned sediemnt is added           
    if depth(cohort) == 0
         tempal_slow(cohort)=dep_slow;
         tempal_fast(cohort)=dep_fast;
         tempal_min(cohort)=dep_min;
    else
         tempal_slow(cohort)=0;
         tempal_fast(cohort)=0;
         tempal_min(cohort)=0;
    end
%Determine depth-dependend decomposition parameters                                   
k_fast(cohort)=ktop_fast(yr)* a_depth.*exp(b_depth.*depth(cohort));
k_slow(cohort)=ktop_slow(yr)* a_depth.*exp(b_depth.*depth(cohort));

    else
         tempbgb_min(cohort)=0;
         tempbgb_fast(cohort) = 0;
         tempbgb_slow(cohort) = 0;
         bgb_org_dep(cohort) = 0;
         
         tempal_slow(cohort)=0;
         tempal_fast(cohort)=0;
         tempal_min(cohort)=0;

         k_fast(cohort)=0;
         k_slow(cohort)=0;
    end
%% Decompose the marsh sediment 
if depth(cohort) <= max_depth_bgb
    decomp_al_fast(cohort) = (al_fast(cohort)+tempal_fast(cohort)).*k_fast(cohort);%Proportion of decomposition in allocthonus fast in each cohort
    al_fast(cohort) = al_fast(cohort)+tempal_fast(cohort)-decomp_al_fast(cohort);%Allocthonus fast in each cohort (total - decomp)    
    decomp_al_slow(cohort) = (al_slow(cohort)+tempal_slow(cohort)).*k_slow(cohort);%Proportion of decomposition in allocthonus slow in each cohort    
    al_slow(cohort) = al_slow(cohort)+tempal_slow(cohort)-decomp_al_slow(cohort);%Allocthonus slow in each cohort (total - decomp)      
    decomp_bgb_fast(cohort) = (bgb_fast(cohort)+tempbgb_fast(cohort)).*k_fast(cohort);%Proportion of decomposition in bgb fast in each cohort     
    bgb_fast(cohort)= bgb_fast(cohort)+tempbgb_fast(cohort)-decomp_bgb_fast(cohort);%bgb fast in each cohort (total - decomp)     
    decomp_bgb_slow(cohort)=(bgb_slow(cohort)+tempbgb_slow(cohort)).*k_slow(cohort);%Proportion of decomposition in bgb slow in each cohort     
    bgb_slow(cohort)= bgb_slow(cohort)+tempbgb_slow(cohort)-decomp_bgb_slow(cohort);%bgb slow in each cohort (total - decomp)    
    al_min(cohort)= al_min(cohort)+tempal_min(cohort);%allocthonous mineral 
    bgb_min(cohort)= bgb_min(cohort)+tempbgb_min(cohort);%bgb mineral
    
    organic(cohort) = al_fast(cohort)+al_slow(cohort)+bgb_fast(cohort)+bgb_slow(cohort);%Organic matter in each cohort
    mineral(cohort) = al_min(cohort)+bgb_min(cohort);%Mineral deposition in each cohort
    loi(cohort) = (organic(cohort)+liveroot(cohort)+liverhizome(cohort))/(organic(cohort)+liveroot(cohort)+liverhizome(cohort)+mineral(cohort));%Loss on Ignition
    
    perc_C(cohort) = .4*loi(cohort)+.0025.*loi(cohort)^2;%Percent carbon in each cohort
    %layer_C(cohort) = (organic(cohort)+liveroot(cohort)+liverhizome(cohort)+mineral(cohort))*perc_C(cohort);%Total carbon
    %layer_C(cohort) = (organic(cohort)+mineral(cohort))*perc_C(cohort);%Total carbon
    layer_C(cohort) = organic(cohort)*perc_C(cohort);%Total carbon

%It's possible under conditions of little to no mineral inputs for layers to be extremely thin - so thin that it falls below matlabs lowest recognized digit (realmin)
%the following keeps this number from being below realmin
    if (organic(cohort)/rhoo)+(mineral(cohort)/rhos) <= realmin*2
        layer_depth(cohort)=realmin;
    else
        layer_depth(cohort) = ((organic(cohort)/rhoo)+(mineral(cohort)/rhos));
    end
end         
end 

    temporgin = sum(bgb_org_dep(1:yr))+dep_fast+dep_slow;%Total organic in
    temporgacc = (sum(organic(1:yr))- organic(yr-1))/rhoo;%Organic accretion     
    tempminacc = (sum(mineral(1:yr))- sum(mineral(1:yr-1)))/rhos;%Mineral accretion  
    tempsoilcolumn = sum(layer_depth(1:yr));%Soil column depth
    tempcolumn_C = sum(layer_C(1:yr));%total Carbon in column
    tempFout_C = sum(decomp_al_fast(1:yr))+sum(decomp_al_slow(1:yr))+sum(decomp_bgb_fast(1:yr))+sum(decomp_bgb_slow(1:yr));%total decomposition
    tempdavgk = mean(k_fast(2500:end));
end