alias bl="byobu ls"
alias bt="byobu attach -t"

function bt1(){ 
    SESSION_NAME=$(byobu ls | head -n 1);
    IFS=':' read -ra arr <<< $SESSION_NAME;
    SESSION_NAME=${arr[0]};
    echo $SESSION_NAME;
    byobu attach -t $SESSION_NAME;
}
function btlast(){ 
    SESSION_NAME=$(byobu ls | tail -n 1);
    IFS=':' read -ra arr <<< $SESSION_NAME;
    SESSION_NAME=${arr[0]};
    echo $SESSION_NAME;
    byobu attach -t $SESSION_NAME;
}


function bk1(){ 
    SESSION_NAME=$(byobu ls | head -n 1);
    IFS=':' read -ra arr <<< $SESSION_NAME;
    SESSION_NAME=${arr[0]};
    echo $SESSION_NAME;
    byobu kill-session -t $SESSION_NAME;
}

function bklast(){ 
    SESSION_NAME=$(byobu ls | tail -n 1);
    IFS=':' read -ra arr <<< $SESSION_NAME;
    SESSION_NAME=${arr[0]};
    echo $SESSION_NAME;
    byobu kill-session -t $SESSION_NAME;
}