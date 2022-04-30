function I_ilumination=ilumination_correction(I)      
    %mmm
    I_fringes = I;
    for i=3:2:100    
        SE = strel('rectangle',[10 i]);
        I_fringes=imopen(I_fringes,SE);
        I_fringes=imclose(I_fringes,SE);
    end
    
    I_ilumination=I-0.9*I_fringes+mean(mean(0.9*I_fringes));
end