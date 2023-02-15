alias bl="byobu ls"
alias bt="byobu attach -t"

function bt1(){ 
    SESSION_NAME=$(byobu ls | head -n 1);
    IFS=':' read -ra arr <<< $SESSION_NAME;
    SESSION_NAME=${arr[0]};
    echo $SESSION_NAME;
    byobu attach -t $SESSION_NAME;
}
