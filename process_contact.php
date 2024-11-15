<?php
if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    // استقبال البيانات من النموذج
    $name = $_POST['name'];
    $email = $_POST['email'];
    $phone = $_POST['phone'];
    $message = $_POST['message'];

    // هنا يمكن إرسال البيانات إلى بريدك الإلكتروني أو قاعدة البيانات
    // على سبيل المثال: 
    // mail("your_email@example.com", "رسالة جديدة من " . $name, $message, "From: " . $email);

    // بعد المعالجة، نقوم بتوجيه المستخدم مع إظهار رسالة نجاح
    header("Location: contact.php?success=1");
    exit();
}
?>
