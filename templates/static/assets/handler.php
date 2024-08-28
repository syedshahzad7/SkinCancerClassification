<?php


    $name = isset($_POST['name']) ? $_POST['name'] : null;
    $email = isset($_POST['email']) ? $_POST['email'] : null;
    $comments = isset($_POST['comments']) ? $_POST['comments'] : null;
    $subject = isset($_POST['subject']) ? $_POST['subject'] : null;
    
    $html = '';
    $html .= 'A message send by: ' .$name . '<br>';
    $html .= 'Email: ' .$email . '<br>';
    $html .= 'Message: '.$comments;
    

    
    if($email){
        if(mail('yourmail@example.com', $subject, $html, $email)){
            echo json_encode(['result' => 'success']);
        }
    } 
?>