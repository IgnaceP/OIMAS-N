function [tempspecies,temproot,temprhizome,tempagb,kk,d,e,D_min,D_max,Ro_T,Rh_T,dWeight] = biomass(Ro_T,Rh_T,npp,Vmax,agb_rh,oN,N_Scalar,coef,yr,d,e,amp,...
    msl,Z,Z_C3,Z_C4)
% Function to calculate biomass related chracteristics - biomass is a function of
% elevation and depth. Includes vegetation switching and weighted
% parameterizations, and root:shoot calculations  

%Updated 05/2019
%% Function Parameters
%Calculates the intersection point of the 2 biomass parabolas 
D_max = amp-[min(Z_C3) min(Z_C4)];%Max and Min depths
D_min = amp-[max(Z_C3) max(Z_C4)];
f1=@(dd) (polyval(coef(1:3),dd));
f2=@(dd) (polyval(coef(4:7),dd));
f = @(dd) f1(dd)-f2(dd);
dd = min(D_min):.0001:max(D_max);
t = f(dd) > 0;
i0 = find(diff(t(:))~=0);
i0 = [i0(:)';i0(:)'+1];
n = size(i0,2);
intersection=zeros(1,1);
for jj = 1:n
    intersection(jj) = fzero(f,dd(i0(:,jj)));    
end

%Defines depth and elevation
d(yr) = amp + msl(yr)-Z(yr);
e(yr) = amp-d(yr);

E_max = [max(Z_C3) max(Z_C4)];%Max and Min elevations
E_min = [min(Z_C3) min(Z_C4)];

%Calculates the aboveground biomass at a given elevation based on biomass parabolas
shootmass = [(polyval(coef(1:3), d(yr))) (polyval(coef(4:7), d(yr)))];
if shootmass(1) < 0
    shootmass(1) = 0;
end
if shootmass(2) < 0
    shootmass(2) = 0;
end

%% Determine habitat (C3, C4, open water, upland)
if  e(yr) > max(E_max) %surface too high for marsh vegetation to grow
    kk = 2;
    tempagb = 0;
    tempspecies = 2;
    temprhizome = 0;
    Root_Shoot = 0;
    dWeight = 0;

elseif e(yr) < min(E_min) %surface is too low for marsh vegetation to grow
    kk = 1;
    tempagb = 0;
    tempspecies = 1;
    temprhizome = 0;
    Root_Shoot = 0;
    dWeight = 0;

elseif e(yr) < max(E_min) && e(yr) >= min(E_min)% flood tolerant species range
    kk=1;%C3 species code
    tempagb = shootmass(kk);%Aboveground biomass
    tempspecies = 1;%C3 species code
    temprhizome=agb_rh(kk)*tempagb;%Rhizome biomass
    Nup = Vmax(kk)*N_Scalar(kk);%N-uptake rate
    Root_Shoot = oN(kk)*npp(kk)/Nup;%Root to Shoot ratio
    dWeight = 0;%Species weight, 0 because 100% C3
elseif e(yr) >= max(E_min) && e(yr) <= amp-intersection
    kk=3;%C3MD species code (C3 Mixed Dominant)
    tempspecies = 3;
    dW = [1 .75 .5];%Defines weights from 50 to 100%
    mix = [max(E_min) .163 amp-intersection];%Defines mixed species range for C3MD - from max elevation to intersection of parabolas
    coefc3 = polyfit((mix),dW,2);%Fit the polynomial that relates species weight to elevation
    dWeight = polyval(coefc3(1:3), e(yr));%Gives the fraction of C3 vegetation in plot 
    
    %Calculates parameters based on weight of dominant species - i.e. weighted averaging of parameters in mixed community
    tempagb = (shootmass(1)*dWeight) + (shootmass(2)*(1-dWeight));
    temprhizome= ((agb_rh(1)*dWeight) + (agb_rh(2)*(1-dWeight)))*tempagb;
    Nup = ((Vmax(1)*dWeight) + (Vmax(2)*(1-dWeight))) * ((N_Scalar(1)*dWeight) + (N_Scalar(2)*(1-dWeight))); 
    Root_Shoot = (oN(kk) * ((npp(1)*dWeight) + (npp(2)*(1-dWeight)))) / Nup;

    Ro_T(3) = (Ro_T(1)*dWeight) + (Ro_T(2)*(1-dWeight));
    Rh_T(3) = (Rh_T(1)*dWeight) + (Rh_T(2)*(1-dWeight));
    
elseif e(yr) >= amp-intersection && e(yr) <= min(E_max)
    kk=4;%C4MD species code (C4 Mixed Dominant)
    tempspecies = 4;%C4MD species code (C4 Mixed Dominant)
    dW = [.5 .75 1];%Defines weights from 50 to 100%
    mix = [amp-intersection .2429 amp-max(D_min)];%Defines mixed species range for C4MD - from intersection of parabolas to max elevation 
    coefc3 = polyfit((mix),dW,2);%Fit the polynomial that relates species weight to elevation
    dWeight = polyval(coefc3(1:3), e(yr));%Gives the fraction of C4 vegetation in plot
    
    %Calculates parameters based on weight of dominant species - i.e. weighted averaging of parameters in mixed community
    tempagb = (shootmass(1)*(1-dWeight)) + (shootmass(2)*dWeight);
    temprhizome= ((agb_rh(1)*(1-dWeight)) + (agb_rh(2)*dWeight))*tempagb;
    Nup = ((Vmax(1)*(1-dWeight)) + (Vmax(2)*dWeight)) * ((N_Scalar(1)*(1-dWeight)) + (N_Scalar(2)*dWeight)); 
    Root_Shoot = (oN(kk) * ((npp(1)*(1-dWeight)) + (npp(2)*dWeight))) / Nup;

    Ro_T(4) = (Ro_T(1)*(1-dWeight)) + (Ro_T(2)*dWeight);
    Rh_T(4) = (Rh_T(1)*(1-dWeight)) + (Rh_T(2)*dWeight);
    
elseif e(yr) > min(E_max) && e(yr) <= max(E_max)   
    kk=2;%C4 species code
    tempspecies = 2;%C4 species code
    tempagb = shootmass(kk);%Aboveground biomass
    temprhizome=agb_rh(kk)*tempagb;%Rhizome biomass
    Nup = Vmax(kk)*N_Scalar(kk);%N-uptake rate
    Root_Shoot = oN(kk)*npp(kk)/Nup; %Root to Shoot ratio 
    dWeight = 0;%Species weight, 0 because 100% C4
end

temproot = Root_Shoot*tempagb;%[g/m^2/yr] Total root biomass in a given year

