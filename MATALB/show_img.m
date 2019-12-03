%%
function reshaped = show_img( A, h, w, d )
    
    if d == 0
        A = A';
        reshaped = reshape(A, h, w);
        figure;
        imagesc(reshaped);
    else
        A = permute(A, [2 1 3]);
        reshaped = reshape(A, h, w, d);
        reshaped = reshaped(:,:,1:3);
        figure;
        imshow(reshaped);
    end
    

end

