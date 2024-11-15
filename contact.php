<!DOCTYPE html>
<html lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>صفحة اتصل بنا</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            direction: rtl;
            text-align: right;
            margin: 20px;
            background-color: #f9f9f9;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }

        table, th, td {
            border: 1px solid #ddd;
        }

        th, td {
            padding: 10px;
            text-align: right;
        }

        th {
            background-color: #f2f2f2;
        }

        .success-message {
            color: green;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }

        .form-container {
            max-width: 600px;
            margin: auto;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }

        button:hover {
            background-color: #45a049;
        }

        input, textarea {
            width: 100%;
            padding: 10px;
            margin-top: 5px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }

        input[type="text"], input[type="email"] {
            height: 40px;
        }

        textarea {
            resize: vertical;
        }

        /* إضافات للهواتف المحمولة */
        @media (max-width: 600px) {
            .form-container {
                width: 100%;
                padding: 15px;
            }
        }
    </style>
</head>
<body>

<div class="form-container">
    <h2>اتصل بنا</h2>

    <!-- رسالة نجاح عند الإرسال -->
    <?php if (isset($_GET['success']) && $_GET['success'] == 1): ?>
        <p class="success-message">تم الإرسال بنجاح!</p>
    <?php endif; ?>

    <!-- نموذج الاتصال -->
    <form action="process_contact.php" method="post">
        <table>
            <tr>
                <th>الاسم الكامل</th>
                <td><input type="text" name="name" required></td>
            </tr>
            <tr>
                <th>البريد الإلكتروني</th>
                <td><input type="email" name="email" required></td>
            </tr>
            <tr>
                <th>رقم الهاتف</th>
                <td><input type="text" name="phone" required></td>
            </tr>
            <tr>
                <th>الرسالة</th>
                <td><textarea name="message" rows="4" required></textarea></td>
            </tr>
        </table>
        <button type="submit">إرسال</button>
    </form>
</div>

</body>
</html>
